from keras.models import Sequential, Model
from keras.layers import Permute, Activation, Dense, Dropout, Embedding, \
        Flatten, Input, merge, Lambda, Reshape, Convolution2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras import backend
from .model_base import MODEL_BASE
import tensorflow as tf
from utils.ngram_nfilter import get_ngram_nfilter

class PACRR(MODEL_BASE):

    params = MODEL_BASE.common_params + ['distill', 'winlen', 'nfilter', 'kmaxpool', 'combine',
            'qproximity', 'context', 'shuffle', 'xfilters', 'cascade']

    def __init__(self, *args, **kwargs):
        super(PACRR, self).__init__(*args, **kwargs)
        self.NGRAM_NFILTER, _ = get_ngram_nfilter(self.p['winlen'], self.p['qproximity'],
                self.p['maxqlen'], self.p['xfilters'])
        self.NGRAMS = sorted(self.NGRAM_NFILTER.keys())
        if self.p['qproximity'] > 0:
            self.NGRAMS.append(self.p['qproximity'])

    def _cascade_poses(self):
        '''
        initialize the cascade positions, over which
        we max-pool after the cnn filters.
        the outcome is a list of document positions.
        when the list only includes the SIM_DIM, it
        is equivalent to max-pool over the whole document
        '''
        doc_poses = list()
        pos_arg = str(self.p['cascade'])
        if len(pos_arg) > 0:
            poses = pos_arg.split('.')
            for p in poses:
                if len(p) > 0:
                    p = int(p)
                    if p <= 0 or p > 100:
                        raise ValueError("Cascade positions are outside (0,100]: %s"%pos_arg)
            doc_poses.extend([int((int(p)/100)*self.p['simdim']) for p in poses if len(p)>0])

        if self.p['simdim'] not in doc_poses:
            doc_poses.append(self.p['simdim'])

        return doc_poses

    def build_doc_scorer(self, r_query_idf, permute_idxs):
        p = self.p
        ng_fsizes = self.NGRAM_NFILTER

        maxpool_poses = self._cascade_poses()

        filter_sizes = list()
        added_fs = set()
        for ng in sorted(ng_fsizes):
            # n-gram in input
            for n_x, n_y in ng_fsizes[ng]:
                dim_name = self._get_dim_name(n_x, n_y)
                if dim_name not in added_fs:
                    filter_sizes.append((n_x,n_y))
                    added_fs.add(dim_name)

        re_input, cov_sim_layers, pool_sdim_layer, pool_sdim_layer_context, pool_filter_layer, ex_filter_layer, re_lq_ds =\
                self._cov_dsim_layers(p['simdim'], p['maxqlen'], filter_sizes, p['nfilter'], top_k=p['kmaxpool'], poses=maxpool_poses, selecter=p['distill'])

        query_idf = Reshape((p['maxqlen'], 1))(Activation('softmax',
            name='softmax_q_idf')(Flatten()(r_query_idf)))


        if p['combine'] < 0:
            raise RuntimeError("combine should be 0 (LSTM) or the number of feedforward dimensions")
        elif p['combine'] == 0:
            rnn_layer = LSTM(1, dropout=0.0, recurrent_regularizer=None, recurrent_dropout=0.0, unit_forget_bias=True, \
                    name="lstm_merge_score_idf", recurrent_activation="hard_sigmoid", bias_regularizer=None, \
                    activation="tanh", recurrent_initializer="orthogonal", kernel_regularizer=None, kernel_initializer="glorot_uniform")

        else:
            dout = Dense(1, name='dense_output')
            d1 = Dense(p['combine'], activation='relu', name='dense_1')
            d2 = Dense(p['combine'], activation='relu', name='dense_2')
            rnn_layer = lambda x: dout(d1(d2(Flatten()(x))))


        def _permute_scores(inputs):
            scores, idxs = inputs
            return tf.gather_nd(scores, backend.cast(idxs, 'int32'))


        self.vis_out = None
        self.visout_count = 0
        def _scorer(doc_inputs, dataid):
            self.visout_count += 1
            self.vis_out = {}
            doc_qts_scores = [query_idf]
            for ng in sorted(ng_fsizes):
                if p['distill'] == 'firstk':
                    input_ng = max(ng_fsizes)
                else:
                    input_ng = ng

                for n_x, n_y in ng_fsizes[ng]:
                    dim_name = self._get_dim_name(n_x, n_y)
                    if n_x == 1 and n_y == 1:
                        doc_cov = doc_inputs[input_ng]
                        re_doc_cov = doc_cov
                    else:
                        doc_cov = cov_sim_layers[dim_name](re_input(doc_inputs[input_ng]))
                        re_doc_cov = re_lq_ds[dim_name](pool_filter_layer[dim_name](Permute((1, 3, 2))(doc_cov)))
                    self.vis_out['conv%s' % ng] = doc_cov

                    if p['context']:
                        ng_signal = pool_sdim_layer_context[dim_name]([re_doc_cov, doc_inputs['context']])
                    else:
                        ng_signal = pool_sdim_layer[dim_name](re_doc_cov)

                    doc_qts_scores.append(ng_signal)

            if len(doc_qts_scores) == 1:
                doc_qts_score = doc_qts_scores[0]
            else:
                doc_qts_score = Concatenate(axis=2)(doc_qts_scores)

            if permute_idxs is not None:
                doc_qts_score = Lambda(_permute_scores)([doc_qts_score, permute_idxs])

            doc_score = rnn_layer(doc_qts_score)
            return doc_score

        return _scorer


    def build_vis(self):
        assert self.visout_count == 1, "cannot vis when _scorer called multiple times (%s)" % self.visout_count

        p = self.p

        doc_inputs = self._create_inputs('doc')
        if p['context']:
            doc_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='doc_context')

        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')
        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=None)

        doc_score = doc_scorer(doc_inputs, 'doc')
        doc_input_list = [doc_inputs[name] for name in doc_inputs]
        visout = [self.vis_out[ng] for ng in sorted(self.vis_out)]
        print("visout:", sorted(self.vis_out))
        self.model = Model(inputs = doc_input_list + [r_query_idf], outputs = [doc_score] + visout)
        return self.model

    def build_predict(self):
        p = self.p

        doc_inputs = self._create_inputs('doc')
        if p['context']:
            doc_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='doc_context')

        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')
        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=None)

        doc_score = doc_scorer(doc_inputs, 'doc')
        doc_input_list = [doc_inputs[name] for name in doc_inputs]
        self.model = Model(inputs = doc_input_list + [r_query_idf], outputs = [doc_score])
        return self.model


    def _create_inputs(self, prefix):
        p = self.p
        if p['distill'] == 'firstk':
            ng = max(self.NGRAMS)
            shared = Input(shape = (p['maxqlen'], p['simdim']), name='%s_wlen_%d' % (prefix, ng))
            inputs = {ng: shared}
        else:
            inputs = {}
            for ng in self.NGRAMS:
                inputs[ng] = Input(shape = (p['maxqlen'], p['simdim']), name='%s_wlen_%d' % (prefix, ng))

        return inputs


    def build(self):
        p = self.p
        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')
        if p['shuffle']:
            permute_input = Input(shape=(p['maxqlen'], 2), name='permute', dtype='int32')
        else:
            permute_input = None

        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=permute_input)

        pos_inputs = self._create_inputs('pos')
        if p['context']:
            pos_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='pos_context')

        pos_score = None
        neg_inputs = {}
        for neg_ind in range(p['numneg']):
            neg_inputs[neg_ind] = self._create_inputs('neg%d' % neg_ind)
            if p['context']:
                neg_inputs[neg_ind]['context'] = Input(shape=(p['maxqlen'], p['simdim']),
                        name='neg%d_context' % neg_ind)

                pos_score = doc_scorer(pos_inputs, 'pos')
        neg_scores = [doc_scorer(neg_inputs[neg_ind], 'neg_%s'%neg_ind) for neg_ind in range(p['numneg'])]

        pos_neg_scores = [pos_score] + neg_scores
        pos_neg_scores = pos_neg_scores[1:] # remove the None from `pos_score = None`
        pos_prob = Lambda(self.pos_softmax, name='pos_softmax_loss')(pos_neg_scores)

        pos_input_list = [pos_inputs[name] for name in pos_inputs]
        neg_input_list = [neg_inputs[neg_ind][ng] for neg_ind in neg_inputs for ng in neg_inputs[neg_ind]]
        inputs = pos_input_list + neg_input_list + [r_query_idf]
        if p['shuffle']:
            inputs.append(permute_input)

        self.model = Model(inputs = inputs, outputs = [pos_prob])

        self.model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
        return self.model
