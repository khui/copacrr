import sys, time, os, itertools, importlib, copy
from utils.common_utils import read_qrel, config_logger
from utils.config import train_test_years, file2name, qrelfdir
from utils.eval_utils import read_run, jud_label, label_jud, year_label_jud, get_epoch_from_val, get_model_param
from utils.year_2_qids import qid_year, year_qids, get_qrelf
import numpy as np, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import logging, warnings

def create_docpairs(qid_cwid_label, test_qids, qid_year, jud_label, label_jud, year_label_jud):
    year_pkey_docpairs = dict()
    for qid in qid_cwid_label:
        if qid not in test_qids:
            continue
        label_cwids = dict()
        year = qid_year[qid]
        if year not in year_pkey_docpairs:
            year_pkey_docpairs[year]=dict()
        for cwid in qid_cwid_label[qid]:
            label = qid_cwid_label[qid][cwid]
            jud = year_label_jud[year][label]
            label = jud_label[jud]
            if label not in label_cwids:
                label_cwids[label] = list()
            label_cwids[label].append(cwid)
        labels = list(label_cwids.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ll, lh = min(labels[i], labels[j]), max(labels[i], labels[j])
                dls, dhs = label_cwids[ll], label_cwids[lh]
                pairkey = '%s-%s'%(label_jud[lh], label_jud[ll])
                if pairkey not in year_pkey_docpairs[year]:
                    year_pkey_docpairs[year][pairkey]=list()
                for dl, dh in itertools.product(dls, dhs):
                    year_pkey_docpairs[year][pairkey].append((qid, dl, dh))
    return year_pkey_docpairs

def eval_docpair_predaccuracy(qid_cwid_score, year_pkey_docpairs, test_year):
    pkey_docpairs = year_pkey_docpairs[test_year]
    pkey_qidcount = dict()
    pkey_qid_acc = dict()
    for pkey in pkey_docpairs:
        qid_dl_dh = pkey_docpairs[pkey]
        if pkey not in pkey_qidcount:
            pkey_qidcount[pkey]=dict()
        for qid, dl, dh in qid_dl_dh:
            if qid not in qid_cwid_score:
                continue
            if qid not in pkey_qidcount[pkey]:
                pkey_qidcount[pkey][qid]=[0,0]
            if dl in qid_cwid_score[qid] and dh in qid_cwid_score[qid]:
                if qid_cwid_score[qid][dl] < qid_cwid_score[qid][dh]:
                    pkey_qidcount[pkey][qid][0]+=1
                pkey_qidcount[pkey][qid][1]+=1
    for pkey in pkey_qidcount:
        pkey_qid_acc[pkey] = dict()
        accs = list()
        total_all = 0
        for qid in pkey_qidcount[pkey]:
            correct, total = pkey_qidcount[pkey][qid]
            total_all += total
            acc = correct / total
            pkey_qid_acc[pkey][qid] = acc
            accs.append(acc)
        pkey_qid_acc[pkey][0] = np.mean(accs)
        pkey_qid_acc[pkey][-1] = total_all
    return pkey_qidcount, pkey_qid_acc


import sacred
from sacred.utils import apply_backspaces_and_linefeeds

ex = sacred.Experiment('metrics')
ex.path = 'metrics' # name of the experiment
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('CUDA_VISIBLE_DEVICES')
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('USER')
ex.captured_out_filter = apply_backspaces_and_linefeeds

from utils.config import default_params
default_params = ex.config(default_params)

@ex.automain
def main(_log, _config):
    p = _config
    modelname = file2name[p['modelfn']]
    mod_model = importlib.import_module('models.%s' % p['modelfn'])
    model_cls = getattr(mod_model, modelname)
    model_params_raw = {k: v for k, v in p.items() if k in model_cls.params or k == 'modelfn'}
    
    list_of_model_params = get_model_param(model_params_raw)
    expids = list()
    for model_params in list_of_model_params:
        model = model_cls(model_params, rnd_seed=p['seed'])
        expid = model.params_to_string(model_params, True)
        expids.append(expid)
    raw_expid = model.params_to_string(model_params_raw, True)
    
    for train_years in train_test_years:

        for i in range(len(train_test_years[train_years])):
            test_year, val_year = train_test_years[train_years][i], train_test_years[train_years][1 - i]
            
            pred_dirs, val_dirs = list(), list()
            
            for expid in expids:
                pred_dir = '%s/train_%s/%s/predict_per_epoch/test_%s/%s' % (p['parentdir'], train_years, p['expname'], test_year, expid)
                val_dir = '%s/train_%s/%s/predict_per_epoch/test_%s/%s' % (p['parentdir'], train_years, p['expname'], val_year, expid)
                if not os.path.isdir(pred_dir) or not os.path.isdir(val_dir):
                    warnings.warn('No such dir {0}/{1}'.format(pred_dir, val_dir), RuntimeWarning)
                    continue
                pred_dirs.append(pred_dir) 
                val_dirs.append(val_dir)
            output_file='%s/train_%s/%s/evaluations/statdocpair/%s_v-%s_t-%s/%s'%(p['outdir'], train_years,\
                    p['expname'], '-'.join(train_years.split('_')), val_year[2:], test_year[2:], raw_expid)

            try:
                if not os.path.isdir(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))
            except OSError as e:
                pass
            _log.info('evaluate {0} on {1} based on val {2} \
                    over docpairs benchmark and output to {3}'.format(expid, test_year, val_year, output_file))

            test_qids = year_qids[test_year]
            qrelf = get_qrelf(qrelfdir, test_year)
            qid_cwid_label = read_qrel(qrelf, test_qids, include_spam=False)
            year_pkey_docpairs = create_docpairs(qid_cwid_label, test_qids, qid_year, jud_label, label_jud, year_label_jud)
            
            best_pred_dir, argmax_epoch, argmax_run, argmax_ndcg, argmax_err = get_epoch_from_val(pred_dirs, val_dirs)

            qid_cwid_invrank, _, runid = read_run(os.path.join(best_pred_dir, argmax_run))
            pkey_qidcount, pkey_qid_acc = eval_docpair_predaccuracy(qid_cwid_invrank, year_pkey_docpairs, test_year)


            dftable = df(pkey_qid_acc, index=sorted(list(qid_cwid_invrank.keys()))+[0, -1])
            _log.info('\n' + dftable.to_string())
            dftable.to_csv(output_file + '.csv', float_format='%.3f', header=True, index=True, sep=',', mode='w')
            _log.info('finished {0} {1} {2} {3}'.format(expid, train_years, val_year, test_year))
