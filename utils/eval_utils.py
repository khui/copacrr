import os, copy
jud_label = {'Nav':4, 'HRel':2, 'Rel':1, 'NRel':0, 'Junk':-2}
label_jud = {4:'Nav', 2:'HRel', 1:'Rel', 0:'NRel', -2:'Junk'}
year_label_jud = {'wt09':{2:'HRel', 1:'Rel', 0:'NRel'},\
                  'wt10':{4:'Nav', 3:'HRel',2:'HRel', 1:'Rel', 0:'NRel', -2:'Junk'},\
                    'wt11':{3:'Nav', 2:'HRel', 1:'Rel', 0:'NRel', -2:'Junk'},\
                  'wt12':{4:'Nav', 3:'HRel', 2:'HRel',1:'Rel', 0:'NRel', -2:'Junk'},\
                  'wt13':{4:'Nav', 3:'HRel', 2:'HRel',1:'Rel', 0:'NRel', -2:'Junk'},\
                  'wt14':{4:'Nav', 3:'HRel',  2:'HRel', 1:'Rel', 0:'NRel', -2:'Junk'}}


def read_run(run_file):
    qid_cwid_score = dict()
    qid_cwid_invrank = dict()
    with open(run_file) as f:
        for line in f:
            cols = line.split()
            qid, cwid, rank, score, runid = int(cols[0]), cols[2], int(cols[3]), float(cols[4]), cols[-1]
            if qid not in qid_cwid_score:
                qid_cwid_score[qid]=dict()
                qid_cwid_invrank[qid]=dict()
            qid_cwid_score[qid][cwid] = score
            qid_cwid_invrank[qid][cwid] = 1 / rank 
    return qid_cwid_invrank, qid_cwid_score, runid

def get_epoch_from_uniqval(test_dir, val_dir):
    def read_all_pred(pred_dir): 
        run_epoch_ndcg_err = dict()
        for run in os.listdir(pred_dir):
            if run.split('.')[-1] != 'run':
                continue
            cols = run[:-4].split('_')
            if len(cols) == 4:
                nb_epoch, ndcg, err, loss = int(cols[0]), float(cols[1]), float(cols[2]), int(cols[3])
                run_epoch_ndcg_err[nb_epoch] = (run, ndcg, err)
        return run_epoch_ndcg_err
    test_epoch_ndcg_err = read_all_pred(test_dir)
    val_epoch_ndcg_err = read_all_pred(val_dir)
    epoch2consider = set(test_epoch_ndcg_err.keys()) & set(val_epoch_ndcg_err.keys())
    # use err for validation
    argmax_epoch = max(epoch2consider, key=lambda e:val_epoch_ndcg_err[e][2])
    argmax_val_run, _, _ = val_epoch_ndcg_err[argmax_epoch]
    argmax_test_run, _, _ = test_epoch_ndcg_err[argmax_epoch]
    return argmax_epoch, argmax_test_run, argmax_val_run

def get_epoch_from_val(test_dirs, val_dirs):
    def read_all_pred(pred_dir): 
        run_epoch_ndcg_err = dict()
        for run in os.listdir(pred_dir):
            if run.split('.')[-1] != 'run':
                continue
            cols = run[:-4].split('_')
            if len(cols) == 5:
                nb_epoch, ndcg, err, loss = int(cols[0]), float(cols[1]), float(cols[-2]), int(cols[-1])
                run_epoch_ndcg_err[nb_epoch] = (run, ndcg, err)
        return run_epoch_ndcg_err
    best_err, best_test_dir, best_epoch = 0, None, 0
    for test_dir, val_dir in zip(test_dirs, val_dirs):
        val_epoch_ndcg_err = read_all_pred(val_dir)
        argmax_epoch = max(val_epoch_ndcg_err, key=lambda e:val_epoch_ndcg_err[e][2])
        _, _, argmax_err = val_epoch_ndcg_err[argmax_epoch]
        if argmax_err > best_err:
            best_test_dir = test_dir
            best_err = argmax_err
            best_epoch = argmax_epoch
    test_epoch_ndcg_err = read_all_pred(best_test_dir)     
    argmax_run, argmax_ndcg, argmax_err = test_epoch_ndcg_err[best_epoch]
    return best_test_dir, best_epoch, argmax_run, argmax_ndcg, argmax_err


def get_model_param(model_params_raw):
    def cast2defaulttype(k, v):
        #TODO we need to convert the param to correct type, now we hard code it.
        if k in ['combine', 'ek']:
            return int(v)
        return v

    # deal with validation for one param
    # for example, the combination could be 32|64|128
    # one would like to generate three model_params or expid for these three
    m_ps = [dict()]
    for k, vs in model_params_raw.items():
        cols = vs.split('|') if type(vs) is str else [vs]
        # with more than one value for a key, 
        # one duplicate the m_ps for the number of values times
        tmp_m_ps = copy.deepcopy(m_ps)
        for i in range(len(cols)-1):
            m_ps.extend(copy.deepcopy(tmp_m_ps))
        for i in range(len(m_ps)):
            m_ps[i][k]=cast2defaulttype(k, cols[i%len(cols)])
    return m_ps
