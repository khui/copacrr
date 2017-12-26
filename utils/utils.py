import logging
import os
import json
import pickle
import h5py
import numpy as np
from collections import Counter
from keras.callbacks import Callback
from keras.utils import plot_model
from . import select_doc_pos
from .config import contextdir


logger = logging.getLogger('pacrr')

class DumpWeight(Callback):
    def __init__(self, weight_dir, batch_size, nb_sample):
        self.weight_dir = weight_dir
        self.batch_size, self.nb_sample = batch_size, nb_sample

    def on_epoch_end(self, epoch, logs={}):
        loss = logs['loss']
        weight_name = '%d_%d_%d_%d.h5'%\
        (epoch, int(loss*10000), self.batch_size, self.nb_sample)
        if not os.path.isdir(self.weight_dir):
            try:
                os.makedirs(self.weight_dir)
            except OSError:
                # sometimes, the dirname is too long, so we have to shorten it
                trimmed_name = '/'.join(self.weight_dir[:-1])
                os.makedirs(trimmed_name)

        weight_file=self.weight_dir + '/' + weight_name
        self.model.save_weights(weight_file)
        logger.info('Callback dumped %s'%weight_name)


def _load_doc_mat_desc(qids, qid_cwid_label, doc_mat_dir, qid_topic_idf, qid_desc_idf, usetopic, usedesc, maxqlen, h5fn=None):
    assert usetopic or usedesc, "must use at least one of topic or desc"

    h5fn = doc_mat_dir + '.hdf5'
    h5 = None
    if h5fn is not None:
        if os.path.isfile(h5fn):
            h5 = h5py.File(h5fn, 'r', libver='latest')

    qid_cwid_simmat = dict()
    qid_term_idf = dict()
    for qid in sorted(qids):
        if qid not in qid_cwid_label:
            logger.error('%d not in qid_cwid_label'%qid)
            continue
        qid_cwid_simmat[qid]=dict()

        topic_idf_arr, desc_idf_arr = qid_topic_idf[qid], qid_desc_idf[qid]
        descmax = maxqlen
        didxs = list(range(len(desc_idf_arr)))
        mi = []
        if usetopic:
            assert len(topic_idf_arr) <= maxqlen, "maxqlen must be >= all topic lens"
            descmax = maxqlen - len(topic_idf_arr)
            mi.append(topic_idf_arr)
        if usedesc:
            if len(didxs) > descmax:
                logger.warning("%s: desc len %s > desc max %s; removing low idf terms" % (qid, len(didxs), descmax))
                didxs = np.sort(np.argsort(desc_idf_arr)[::-1][:descmax])
                logger.info("%s -> %s" % (desc_idf_arr, desc_idf_arr[didxs]))
            mi.append(desc_idf_arr[didxs])
        qid_term_idf[qid] = np.concatenate(mi, axis=0).astype(np.float32)

        if h5 is not None:
            docmap_d = json.loads(h5['/desc/%s' % qid].attrs['docmap'])
            docmap_t = json.loads(h5['/topic/%s' % qid].attrs['docmap'])

        for cwid in qid_cwid_label[qid]:
            topic_cwid_f = doc_mat_dir + '/topic_doc_mat/%d/%s.npy'%(qid, cwid)
            desc_cwid_f = doc_mat_dir + '/desc_doc_mat/%d/%s.npy'%(qid, cwid)
            topic_mat, desc_mat = np.empty((0,0), dtype=np.float32), np.empty((0,0), dtype=np.float32)
            if h5 is not None and cwid not in docmap_t:
                logger.error('topic %s not exist.'%cwid)
            elif h5 is None and not os.path.isfile(topic_cwid_f):
                logger.warning('%s not exist.'%topic_cwid_f)
            elif usetopic:
                if h5 is None:
                    topic_mat = np.load(topic_cwid_f)
                else:
                    topic_mat = np.vstack(h5['/topic/%s' % qid][docmap_t[cwid]])
                if len(topic_mat.shape) != 2:
                    logger.warning('topic_mat {0} {1} {2}'.format(qid, cwid, topic_mat.shape))
                    continue
            if h5 is not None and cwid not in docmap_d:
                logger.error('desc %s not exist.'%cwid)
            elif h5 is None and not os.path.isfile(desc_cwid_f):
                logger.warning('%s not exist.'%desc_cwid_f)
            elif usedesc:
                if h5 is None:
                    desc_mat = np.load(desc_cwid_f)[didxs]
                else:
                    desc_mat = np.vstack(h5['/desc/%s' % qid][docmap_d[cwid]])[didxs]
                if len(desc_mat.shape) != 2:
                    logger.warning('desc_mat {0} {1} {2}'.format(qid, cwid, desc_mat.shape))
                    continue
            #if topic_mat.shape[1] == desc_mat.shape[1] and topic_mat.shape[1]>0:
            empty = True
            m = []
            if usetopic:
                m.append(topic_mat)
                if topic_mat.shape[1] > 0:
                    empty = False
            if usedesc:
                m.append(desc_mat)
                if desc_mat.shape[1] > 0:
                    empty = False
            if usetopic and usedesc and topic_mat.shape[1] != desc_mat.shape[1]:
                empty = True
            if not empty:
                qid_cwid_simmat[qid][cwid] = np.concatenate(m, axis=0).astype(np.float32)

            else:
                logger.warning('dimension mismatch {0} {1} {2} {3}'.format(qid, cwid, topic_mat.shape, desc_mat.shape))

    if h5 is not None:
        h5.close()

    return qid_cwid_simmat, qid_term_idf

def load_query_idf(qids, doc_mat_dir):
    idfdir_topic = doc_mat_dir + '/query_idf/topic_term_idf'
    idfdir_desc = doc_mat_dir + '/query_idf/desc_term_idf'
    qid_topic_idf = dict()
    qid_desc_idf = dict()
    for qid in qids:
        if not os.path.isfile(idfdir_topic + '/%d.npy'%qid) or not os.path.isfile(idfdir_desc + '/%d.npy'%qid):
            logger.error('%d in %s or %s not exist'%(qid, idfdir_topic,idfdir_desc))
            continue
        qid_topic_idf[qid] = np.load(idfdir_topic + '/%d.npy'%qid)
        qid_desc_idf[qid] = np.load(idfdir_desc + '/%d.npy'%qid)
    return qid_topic_idf, qid_desc_idf



def convert_cwid_udim_simmat(qids, qid_cwid_rmat, select_pos_func, \
        qid_term_idf, qid_cwid_qermat,\
                             dim_sim, max_query_term, n_grams, context):
    qid_cwid_mat=dict()
    qid_context = {}
    qid_ext_idfarr = dict()
    pad_value = 0
    for qid in qids:
        if qid not in qid_cwid_rmat:
            logger.error('%d not in qid_cwid_rmat'%qid)
            continue
        if len(qid_cwid_rmat[qid]) == 0:
            logger.error('%d includes 0 docs'%qid)
            continue

        qid_cwid_mat[qid]=dict()
        if context:
            qid_context_raw = {k: np.array(v, dtype=np.float32) for k, v in
                            pickle.load(open('%s/%s.p' % (contextdir, qid), 'rb')).items()}
            qid_context[qid] = {}
        for cwid in qid_cwid_rmat[qid]:
            len_doc = qid_cwid_rmat[qid][cwid].shape[1]
            len_query = qid_cwid_rmat[qid][cwid].shape[0]
            if qid not in qid_ext_idfarr:
                qid_ext_idfarr[qid] =  np.pad(qid_term_idf[qid],\
                        pad_width=((0,max_query_term-len_query)),\
                                      mode='constant', constant_values=-np.inf)
            for n_gram in n_grams:
                if n_gram not in qid_cwid_mat[qid]:
                    qid_cwid_mat[qid][n_gram]=dict()
                if len_doc > dim_sim:
                    rmat = np.pad(qid_cwid_rmat[qid][cwid],  pad_width=((0,max_query_term-len_query),(0, 1)), mode='constant', constant_values=pad_value).astype(np.float32)
                    selected_inds = select_pos_func(qid_cwid_rmat[qid][cwid], dim_sim, n_gram)

                    if qid_cwid_qermat is None:
                        qid_cwid_mat[qid][n_gram][cwid] = (rmat[:, selected_inds])
                    else:
                        qermat = np.pad(qid_cwid_qermat[qid][cwid],  pad_width=((0,max_query_term-len_query),(0, 1)), mode='constant', constant_values=pad_value).astype(np.float32)
                        qid_cwid_mat[qid][n_gram][cwid] = (rmat[:, selected_inds],\
                                qermat[:, selected_inds])

                    if context:
                        qid_context[qid][cwid] = qid_context_raw[cwid][selected_inds]
                elif len_doc < dim_sim:
                    if qid_cwid_qermat is None:
                        qid_cwid_mat[qid][n_gram][cwid] = \
                                (np.pad(qid_cwid_rmat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),\
                                (0, dim_sim - len_doc)),mode='constant', \
                                constant_values=pad_value).astype(np.float32))
                    else:
                        qid_cwid_mat[qid][n_gram][cwid] = \
                                (np.pad(qid_cwid_rmat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),\
                                (0, dim_sim - len_doc)),mode='constant', \
                                constant_values=pad_value).astype(np.float32),\
                                np.pad(qid_cwid_qermat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),\
                                (0, dim_sim - len_doc)),mode='constant', \
                                constant_values=pad_value).astype(np.float32))

                    if context:
                        qid_context[qid][cwid] = np.pad(qid_context_raw[cwid],
                                                        pad_width=((0, dim_sim - len_doc),),
                                                        mode='constant', constant_values=pad_value)

                else:
                    if qid_cwid_qermat is None:
                        qid_cwid_mat[qid][n_gram][cwid] = (np.pad(qid_cwid_rmat[qid][cwid],\
                                        pad_width=((0,max_query_term-len_query),(0, 0)),\
                                mode='constant', constant_values=pad_value).astype(np.float32))
                    else:
                        qid_cwid_mat[qid][n_gram][cwid] = (np.pad(qid_cwid_rmat[qid][cwid],\
                                        pad_width=((0,max_query_term-len_query),(0, 0)),\
                                mode='constant', constant_values=pad_value).astype(np.float32),\
                                np.pad(qid_cwid_qermat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),(0, 0)),\
                                mode='constant', constant_values=pad_value).astype(np.float32))
                    if context:
                        qid_context[qid][cwid] = qid_context_raw[cwid]

                # hack so that we have the same shape as the sim matrices
                if context:
                    qid_context[qid][cwid] = np.array([qid_context[qid][cwid] for i in range(max_query_term)], dtype=np.float32)
                else:
                    qid_context = None

    return qid_cwid_mat, qid_ext_idfarr, qid_context



def sample_train_data_weighted(qid_wlen_cwid_mat, qid_cwid_label, \
        query_idfs, sample_qids, binarysimm, \
            label2tlabel={4:2,3:2,2:2,1:1,0:0,-2:0},\
            sample_label_prob={2:0.5,1:0.5},\
            n_query_terms = 16,\
            NUM_NEG=10,\
            n_dims = 300, n_batch=32, random_shuffle=True, random_seed=14, qid_context=None):
    np.random.seed(random_seed)
    context = qid_context is not None
    qid_label_cwids=dict()
    label_count = dict()
    label_qid_count = dict()
    for qid in sample_qids:
        if qid not in qid_cwid_label or qid not in qid_wlen_cwid_mat:
            logger.error('%d in qid_cwid_label %r, in qid_cwid_mat %r'%\
                    (qid,qid in qid_cwid_label, qid in qid_wlen_cwid_mat))
            continue
        qid_label_cwids[qid]=dict()
        wlen_k = list(qid_wlen_cwid_mat[qid].keys())[0]
        for cwid in qid_cwid_label[qid]:
            l = label2tlabel[qid_cwid_label[qid][cwid]]
            if cwid not in qid_wlen_cwid_mat[qid][wlen_k]:
                logger.error('%s not in %d in qid_wlen_cwid_mat'%(cwid, qid))
                continue
            if l not in qid_label_cwids[qid]:
                qid_label_cwids[qid][l] = list()
            qid_label_cwids[qid][l].append(cwid)
            if l not in label_qid_count:
                label_qid_count[l] = dict()
            if qid not in label_qid_count[l]:
                label_qid_count[l][qid]=0
            label_qid_count[l][qid] += 1
            if l not in label_count:
                label_count[l] = 0
            label_count[l] += 1

    if len(sample_label_prob) == 0:
        total_count = sum([label_count[l] for l in label_count if l > 0])
        sample_label_prob = {l:label_count[l]/float(total_count) for l in label_count if l > 0}
        logger.error('nature sample_label_prob', sample_label_prob)
    label_qid_prob = dict()
    for l in label_qid_count:
        if l > 0:
            total_count = label_count[l]
            label_qid_prob[l] = {qid:label_qid_count[l][qid]/float(total_count) for qid in label_qid_count[l]}
    sample_label_qid_prob = {l:[label_qid_prob[l][qid] if qid in label_qid_prob[l] else 0 for qid in sample_qids] \
            for l in label_qid_prob}
    while 1:
        pos_batch = dict()
        neg_batch = dict()
        qid_batch = list()
        pcwid_batch = list()
        ncwid_batch = list()
        qidf_batch = list()
        pos_context_batch = []
        neg_context_batch = {}
        ys = list()
        selected_labels = np.random.choice([l for l in sorted(sample_label_prob)], \
                size=n_batch, replace=True, p=[sample_label_prob[l] for l in sorted(sample_label_prob)])
        label_counter = Counter(selected_labels)
        total_train_num = 0
        for label in label_counter:
            nl_selected = label_counter[label]
            if nl_selected == 0:
                continue
            selected_qids = np.random.choice(sample_qids, \
                    size=nl_selected, replace=True, p=sample_label_qid_prob[label])
            qid_counter = Counter(selected_qids)
            for qid in qid_counter:
                pos_label = 0
                nq_selected = qid_counter[qid]
                if nq_selected == 0:
                    continue
                for nl in reversed(range(label)):
                    if nl in qid_label_cwids[qid]:
                        pos_label = label
                        neg_label = nl
                        break
                if pos_label != label:
                    continue
                pos_cwids = qid_label_cwids[qid][label]
                neg_cwids = qid_label_cwids[qid][nl]
                n_pos, n_neg = len(pos_cwids), len(neg_cwids)
                idx_poses = np.random.choice(list(range(n_pos)),size=nq_selected, replace=True)
                min_wlen = min(qid_wlen_cwid_mat[qid])
                for wlen in qid_wlen_cwid_mat[qid]:
                    if wlen not in pos_batch:
                        pos_batch[wlen] = list()
                    for pi in idx_poses:
                        p_cwid = pos_cwids[pi]
                        pos_batch[wlen].append(qid_wlen_cwid_mat[qid][wlen][p_cwid])
                        if wlen == min_wlen:
                            if context:
                                pos_context_batch.append(qid_context[qid][p_cwid])
                            ys.append(1)
                for neg_ind in range(NUM_NEG):
                    idx_negs = np.random.choice(list(range(n_neg)),size=nq_selected, replace=True)
                    min_wlen = min(qid_wlen_cwid_mat[qid])
                    for wlen in qid_wlen_cwid_mat[qid]:
                        if wlen not in neg_batch:
                            neg_batch[wlen] = dict()
                        if neg_ind not in neg_batch[wlen]:
                            neg_batch[wlen][neg_ind]=list()
                            if wlen == min_wlen and context:
                                neg_context_batch[neg_ind] = []
                        for ni in idx_negs:
                            n_cwid = neg_cwids[ni]
                            neg_batch[wlen][neg_ind].append(qid_wlen_cwid_mat[qid][wlen][n_cwid])
                            if wlen == min_wlen and context:
                                neg_context_batch[neg_ind].append(qid_context[qid][n_cwid])
                qidf_batch.append(query_idfs[qid].reshape((1,n_query_terms,1)).repeat(nq_selected, axis=0))
        total_train_num = len(ys)
        if random_shuffle:
            shuffled_index=np.random.permutation(list(range(total_train_num)))
        else:
            shuffled_index = list(range(total_train_num))
        train_data = dict()
        labels = np.array(ys)[shuffled_index]

        getmat = lambda x: np.array(x)

        for wlen in pos_batch:
            train_data['pos_wlen_%d'%wlen] = getmat(pos_batch[wlen])[shuffled_index,:]
            for neg_ind in range(NUM_NEG):
                train_data['neg%d_wlen_%d'%(neg_ind,wlen)] = np.array(getmat(neg_batch[wlen][neg_ind]))[shuffled_index,:]

        if binarysimm:
            for k in train_data:
                assert k.find("_wlen_") != -1, "data contains non-simmat objects"
                train_data[k] = (train_data[k] >= 0.999).astype(np.int8)

        if context:
            train_data['pos_context'] = np.array(pos_context_batch)[shuffled_index]
            for neg_ind in range(NUM_NEG):
                train_data['neg%d_context' % neg_ind] = np.array(neg_context_batch[neg_ind])[shuffled_index]

        train_data['query_idf'] = np.concatenate(qidf_batch, axis=0)[shuffled_index,:]

        train_data['permute'] = np.array([[(bi, qi) for qi in np.random.permutation(n_query_terms)]
                                          for bi in range(n_batch)], dtype=np.int)
        yield (train_data, labels)

def dump_modelplot(model, model_file):
    try:
        plot_model(model, to_file=model_file + '.pdf',show_shapes=True)

    # we try to include all the model params in the name of the output file
    # but that sometimes makes the name too long
    # as a simple solution, we fallback on the much shorter 'model.pdf' if
    # the original name is too long
    except IOError:
        model_file_path = "/".join(model_file.split('/')[:-1]) + '/'
        plot_model(model, to_file=model_file_path + 'model.pdf',show_shapes=True)

def pred_label(model, input_x, input_cwid, input_qid):
    batch_size = min(len(input_cwid), 256)
    preds = model.predict(input_x, batch_size=batch_size, verbose=0).ravel()
    qid_cwid_pred = dict()
    for qid, cwid, pred in zip(input_qid, input_cwid, preds.tolist()):
        if qid not in qid_cwid_pred:
            qid_cwid_pred[qid] = dict()
        qid_cwid_pred[qid][cwid] = pred
    return qid_cwid_pred



def load_test_data(qids, rawdoc_mat_dir, qid_cwid_label, N_GRAMS, param_val):
    POS_METHOD = param_val['distill']
    SIM_DIM = param_val['simdim']
    NUM_NEG = param_val['numneg']
    MAX_QUERY_LENGTH = param_val['maxqlen']
    binarysimm = param_val['binmat']
    CONTEXT = param_val['context']
    if POS_METHOD == 'firstk':
        mat_ngrams = [max(N_GRAMS)]
    else:
        mat_ngrams = N_GRAMS

    select_pos_func = getattr(select_doc_pos, 'select_pos_%s'%POS_METHOD)
    qid_topic_idf, qid_desc_idf = load_query_idf(qids, rawdoc_mat_dir)
    qid_cwid_rmat, qid_term_idf = _load_doc_mat_desc(qids, qid_cwid_label, rawdoc_mat_dir, qid_topic_idf, \
            qid_desc_idf, usetopic=param_val['ut'], usedesc=param_val['ud'], maxqlen=MAX_QUERY_LENGTH)

    qid_cwid_rqexpmat = None
    qid_cwid_mat, qid_ext_idfarr, qid_context = convert_cwid_udim_simmat(qids, qid_cwid_rmat, select_pos_func, qid_term_idf, qid_cwid_rqexpmat,
                                                                dim_sim=SIM_DIM, max_query_term=MAX_QUERY_LENGTH,n_grams=mat_ngrams, context=CONTEXT)

    doc_vec, q_idfs, cwids, testqids  = dict(), list(), list(), list()
    contexts = []
    for qid in qids:
        if qid not in qid_cwid_label or qid not in qid_cwid_mat:
            logger.error('Error: %d not in qid_cwid_label or not in qid_cwid_mat for test data'%qid)
            continue
        if len(qid_cwid_mat[qid]) == 0:
            logger.error('Error: no doc in qid_cwid_mat for %d'%qid)
            continue
        cwid_label = qid_cwid_label[qid]
        min_wlen = min(qid_cwid_mat[qid])
        for wlen in qid_cwid_mat[qid]:
            if wlen not in doc_vec:
                doc_vec[wlen] = list()
            for cwid in qid_cwid_mat[qid][wlen]:
                if wlen == min_wlen:
                    cwids.append(cwid)
                    testqids.append(qid)
                    q_idfs.append(qid_ext_idfarr[qid].reshape((1, qid_ext_idfarr[qid].shape[0],1)))
                    if CONTEXT:
                        contexts.append(qid_context[qid][cwid])
                doc_vec[wlen].append(qid_cwid_mat[qid][wlen][cwid])

    getmat = lambda x: np.array(x)

    test_data = {'doc_wlen_%d'%wlen: np.array(getmat(doc_vec[wlen])) for wlen in doc_vec}

    if binarysimm:
        for k in test_data:
            assert k.find("_wlen_") != -1, "data contains non-simmat objects"
            test_data[k] = (test_data[k] >= 0.999).astype(np.int8)

    q_idfs = np.concatenate(q_idfs, axis=0)
    logger.info('Test data: {0} {1} {2}'.format([(wlen, getmat(doc_vec[wlen]).shape) for wlen in doc_vec], q_idfs.shape, len(cwids)))
    test_data['query_idf']=q_idfs

    if CONTEXT:
        test_data['doc_context'] = np.array(contexts)
    return test_data, cwids, testqids

def load_train_data_generator(qids, rawdoc_mat_dir, qid_cwid_label, N_GRAMS, param_val,\
        label2tlabel={4:2,3:2,2:2,1:1,0:0,-2:0}, sample_label_prob={2:0.5,1:0.5}):
    POS_METHOD = param_val['distill']
    SIM_DIM = param_val['simdim']
    NUM_NEG = param_val['numneg']
    MAX_QUERY_LENGTH = param_val['maxqlen']
    binarysimm = param_val['binmat']
    CONTEXT = param_val['context']
    n_batch = param_val['batch']
    rnd_seed = param_val['seed']
    if POS_METHOD == 'firstk':
        mat_ngrams = [max(N_GRAMS)]
    else:
        mat_ngrams = N_GRAMS
    assert POS_METHOD == 'firstk' or not CONTEXT, "context is misaligned if we aren't using firstk"

    if 'cw' in param_val and param_val['modelfn'] == 'cw_pacrr':
        global contextdir
        contextdir = os.path.join(contextdir, 'win_%s' % param_val['cw'])
        if not os.path.exists(contextdir):
            raise RuntimeError("missing dir for cw=%s: %s" % (param_val['cw'], contextdir))

    select_pos_func = getattr(select_doc_pos, 'select_pos_%s'%POS_METHOD)
    qid_topic_idf, qid_desc_idf = load_query_idf(qids, rawdoc_mat_dir)
    qid_cwid_rmat, qid_term_idf = _load_doc_mat_desc(qids, qid_cwid_label, rawdoc_mat_dir, qid_topic_idf, qid_desc_idf, usetopic=param_val['ut'], usedesc=param_val['ud'], maxqlen=MAX_QUERY_LENGTH)
    qid_cwid_rqexpmat = None
    qid_wlen_cwid_mat, qid_ext_idfarr, qid_context = convert_cwid_udim_simmat(qids, qid_cwid_rmat, select_pos_func, qid_term_idf, qid_cwid_rqexpmat,\
            dim_sim=SIM_DIM, max_query_term=MAX_QUERY_LENGTH, n_grams=mat_ngrams, context=CONTEXT)
    train_data_generator=\
    sample_train_data_weighted(qid_wlen_cwid_mat, qid_cwid_label, qid_ext_idfarr, qids, \
            binarysimm=binarysimm, label2tlabel=label2tlabel,\
            sample_label_prob=sample_label_prob,\
            n_query_terms = MAX_QUERY_LENGTH, NUM_NEG=NUM_NEG,\
            n_dims = SIM_DIM, n_batch=n_batch, random_shuffle=True, random_seed=rnd_seed, qid_context=qid_context)
    return train_data_generator


