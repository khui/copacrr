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

import sacred
from sacred.utils import apply_backspaces_and_linefeeds

ex = sacred.Experiment('metrics')
ex.path = 'metrics' # name of the experiment
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('CUDA_VISIBLE_DEVICES')
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('USER')
ex.captured_out_filter = apply_backspaces_and_linefeeds

from utils.config import default_params
default_params = ex.config(default_params)


def create_docpairs(qid_cwid_label, test_qids, qid_year):
    docpairs = {}
    for qid in qid_cwid_label:
        assert qid in test_qids
        year = qid_year[qid]
        docpairs.setdefault(year, {})
        
        label_cwids = {}
        for cwid, raw_label in qid_cwid_label[qid].items():
            jud = year_label_jud[year][raw_label]
            label = jud_label[jud]
            label_cwids.setdefault(label, []).append(cwid)
           
        for l1, l2 in itertools.combinations(sorted(label_cwids), 2):
            pairkey = "%s-%s" % (label_jud[l2], label_jud[l1])
            docpairs[year].setdefault(pairkey, [])

            for dl, dh in itertools.product(label_cwids[l1], label_cwids[l2]):
                docpairs[year][pairkey].append((qid, dl, dh))
                
    return docpairs

    
def eval_docpair_predaccuracy(qid_cwid_score, docpairs, test_year):
    pkey_qidcount = dict()
    for pkey, docpairs in docpairs[test_year].items():
        for qid, dl, dh in docpairs:
            #TODO why is this necessary?
            if qid not in qid_cwid_score:
                continue

            pkey_qidcount.setdefault(pkey, {}).setdefault(qid, [])
            #TODO better way to handle missing simmats?
            if dl in qid_cwid_score[qid] and dh in qid_cwid_score[qid]:
                pkey_qidcount[pkey][qid].append(qid_cwid_score[qid][dl] < qid_cwid_score[qid][dh])

    pkey_qid_acc = {}
    for pkey, qids in pkey_qidcount.items():
        for qid, outcomes in qids.items():
            correct, total = sum(outcomes), len(outcomes)
            pkey_qid_acc.setdefault(pkey, {})[qid] = float(correct) / total

        pkey_qid_acc[pkey][0] = np.mean(list(pkey_qid_acc[pkey].values()))
        pkey_qid_acc[pkey][-1] = sum(len(outcomes) for outcomes in qids.values())
        
    return pkey_qid_acc


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
            year_pkey_docpairs = create_docpairs(qid_cwid_label, test_qids, qid_year)
            
            best_pred_dir, argmax_epoch, argmax_run, argmax_ndcg, argmax_err = get_epoch_from_val(pred_dirs, val_dirs)

            qid_cwid_invrank, _, runid = read_run(os.path.join(best_pred_dir, argmax_run))
            pkey_qid_acc = eval_docpair_predaccuracy(qid_cwid_invrank, year_pkey_docpairs, test_year)


            dftable = df(pkey_qid_acc, index=sorted(list(qid_cwid_invrank.keys()))+[0, -1])
            _log.info('\n' + dftable.to_string())
            dftable.to_csv(output_file + '.csv', float_format='%.3f', header=True, index=True, sep=',', mode='w')
            _log.info('finished {0} {1} {2} {3}'.format(expid, train_years, val_year, test_year))
