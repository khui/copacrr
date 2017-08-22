from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Permute, Dense, Flatten, Input, Lambda, Reshape
from keras import backend
import tensorflow as tf
backend.clear_session()

# map the file name of the model_file_name.py to the model class name
file2name={'pacrr_ce':'PACRR'}
name2file = {v: k for k, v in file2name.items()}

_boolstr = lambda x: x.lower() == 'true'
_nonestr = lambda x: None if x == 'None' else x
param_types = {'distill': str, 'xfilters': str, 'cascade': str,
               'context': _boolstr, 'binmat': _boolstr, 'shuffle': _boolstr,
               'ut': _boolstr, 'ud': _boolstr, 'enhance': _nonestr}


class MODEL_BASE:
    common_params = ['simdim', 'epochs', 'nsamples', 'maxqlen', 'binmat', 'numneg', 'batch', 'ud', 'ut']

    def params_to_string(self, params, skip_check=False):
        s = [file2name[params['modelfn']]]

        # force cascade's type because some values are ambiguous
        if 'cascade' in params:
            params['cascade'] = str(params['cascade'])
        
        for k in self.params:
            # don't include ut/ud when true (hack to prevent previous expnames from changing)
            if (k == "ut" or k == "ud") and params[k]:
                continue
            s.append("%s-%s" % (k, params[k]))

        s = "_".join(s)

        if not skip_check:
            if params != self.string_to_params(s, True):
                d = self.string_to_params(s, True)
                for k, v in d.items():
                    if k not in params or params.get(k) != v:
                        print("%s k=%s vs. k=%s" % (k, params.get(k), d[k]))
                for k, v in params.items():
                    if k not in d or d.get(k) != v:
                        print("%s k=%s vs. k=%s" % (k, params[k], d.get(k)))
                raise RuntimeError("self.string_to_params(s, True)")
        return s


    def string_to_params(self, s, skip_check=False):
        fields = s.split('_')
        modelname = fields[0]
        params = fields[1:]

        out = {'modelfn': name2file[modelname]}
        for pstr in params:
            k, v = pstr.split("-")
            if (k == "ut" or k == "ud") and _boolstr(v):
                continue
            assert k in self.params, "invalid key '%s' encountered in string: %s" % (k, s)
            assert k not in out, "duplicate key '%s' in string: %s" % (k, s)
            out[k] = param_types.get(k, int)(v)

        # assume ut/ud true if missing (hack to prevent previous expnames from changing)
        if 'ut' not in out:
            out['ut'] = True
        if 'ud' not in out:
            out['ud'] = True

        if not skip_check:
            assert s == self.params_to_string(out, True), "asymmetric string_to_params on string: %s" % s
        return out

    def __init__(self, p, rnd_seed):
        self.p = p
        self.rnd_seed = rnd_seed
    
    def pos_softmax(self, pos_neg_scores):
        exp_pos_neg_scores = [tf.exp(s) for s in pos_neg_scores]
        denominator = tf.add_n(exp_pos_neg_scores)
        return exp_pos_neg_scores[0] / denominator

    def _kmax(self, x, top_k):
        return tf.nn.top_k(x, k=top_k, sorted=True, name=None)[0]

    def _kmax_context(self, inputs, top_k):
        x, context_input = inputs
        vals, idxs = tf.nn.top_k(x, k=top_k, sorted=True)
        shape = tf.shape(x)
        mg = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape[:(x.get_shape().ndims - 1)]) + [top_k])], indexing='ij')
        val_contexts = tf.gather_nd(context_input, tf.stack(mg[:-1] + [idxs], axis=-1))
    
    def _multi_kmax_concat(self, x, top_k, poses):
        slice_mats=list()
        for p in poses:
            slice_mats.append(tf.nn.top_k(tf.slice(x, [0,0,0], [-1,-1,p]), k=top_k, sorted=True, name=None)[0])
        concat_topk_max = tf.concat(slice_mats, -1, name='concat')
        return concat_topk_max

    def _multi_kmax_context_concat(self, inputs, top_k, poses):
        x, context_input = inputs
        idxes, topk_vs = list(), list()
        for p in poses:
            val, idx = tf.nn.top_k(tf.slice(x, [0,0,0], [-1,-1, p]), k=top_k, sorted=True, name=None)
            topk_vs.append(val)
            idxes.append(idx)
        concat_topk_max = tf.concat(topk_vs, -1, name='concat_val')
        concat_topk_idx = tf.concat(idxes, -1, name='concat_idx')
        shape = tf.shape(x)
        mg = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape[:(x.get_shape().ndims - 1)]) + [top_k*len(poses)])], indexing='ij')
        val_contexts = tf.gather_nd(context_input, tf.stack(mg[:-1] + [concat_topk_idx], axis=-1))

        return backend.concatenate([concat_topk_max, val_contexts])

    def _get_dim_name(self, n_x, n_y):
        return '%dx%d'%(n_x,n_y)

    def _cov_dsim_layers(self, dim_sim, len_query, n_grams, n_filter, top_k, poses, selecter):
        re_input = Reshape((len_query, dim_sim, 1), name='ql_ds_doc')
        cov_sim_layers = dict()
        pool_sdim_layer=dict()
        pool_sdim_layer_context=dict()
        re_ql_ds=dict()
        pool_filter_layer=dict()
        for n_query, n_doc in n_grams:
            subsample_docdim = 1
            if selecter in ['strides']:
                subsample_docdim = n_doc
            dim_name = self._get_dim_name(n_query,n_doc)
            cov_sim_layers[dim_name] = \
            Conv2D(n_filter, kernel_size=(n_query, n_doc), strides=(1, subsample_docdim), padding="same", use_bias=True,\
                    name='cov_doc_%s'%dim_name, kernel_initializer='glorot_uniform', activation='relu', \
                    bias_constraint=None, kernel_constraint=None, data_format=None, bias_regularizer=None,
                    activity_regularizer=None, weights=None, kernel_regularizer=None)

            pool_sdim_layer[dim_name] = Lambda(lambda x: self._multi_kmax_concat(x, top_k, poses), \
                    name='ng_max_pool_%s_top%d_pos%d'%(dim_name, top_k, len(poses)))
            pool_sdim_layer_context[dim_name] = \
                    Lambda(lambda x: self._multi_kmax_context_concat(x, top_k, poses), \
                    name='ng_max_pool_%s_top%d__pos%d_context'%(dim_name, top_k, len(poses))) 
            re_ql_ds[dim_name] = Lambda(lambda t:backend.squeeze(t,axis=2), name='re_ql_ds_%s'%(dim_name))
            pool_filter_layer[dim_name] = \
                    MaxPooling2D(pool_size=(1, n_filter), strides=None, padding='valid', data_format=None, \
                    name='max_over_filter_doc_%s'%dim_name)

        ex_filter_layer = Permute((1, 3, 2), input_shape=(len_query, 1, n_filter), name='permute_filter_lenquery') 
        return re_input, cov_sim_layers, pool_sdim_layer, pool_sdim_layer_context, pool_filter_layer, ex_filter_layer, re_ql_ds

    def dump_weights(self, weight_file):
        self.model.save_weights(weight_file)
        
    def build_from_dump(self, weight_file):
        self.build_predict()
        self.model.load_weights(weight_file)
        return self.model
