import sys, time, os, itertools, shutil, subprocess, time, copy, importlib
from scipy.stats import ttest_rel, ttest_ind
from utils.common_utils import read_qrel, config_logger
from utils.config import train_test_years, file2name, perlf, trec_run_basedir, eval_trec_run_basedir, qrelfdir
from utils.eval_utils import read_run, jud_label, label_jud, year_label_jud, get_epoch_from_val, get_model_param
from utils.year_2_qids import qid_year, year_qids, get_qrelf
import numpy as np, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import logging, warnings



def trec_run_predscore(_log, trec_run_dir, qid_cwid_pred):
    sysn_qid_cwid_score = dict()
    for f in os.listdir(trec_run_dir):
        _, qid_cwid_score, trecsys_name = read_run(os.path.join(trec_run_dir, f))
        for qid in qid_cwid_score:
            if qid not in qid_cwid_pred:
                _log.error('{0} is not included qid_cwid_pred'.format((qid)))
                continue
            for cwid in qid_cwid_score[qid]:
                if cwid in qid_cwid_pred[qid]:
                    qid_cwid_score[qid][cwid] = qid_cwid_pred[qid][cwid]
                else:
                    qid_cwid_score[qid][cwid]=-float('Inf')
        sysn_qid_cwid_score[trecsys_name] = dict(qid_cwid_score)
    return sysn_qid_cwid_score

def print_rerun(sysn_qid_cwid_pred, outdir, pred_expid, val_year, test_year):
    try:
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
    except OSError as e:
        pass
    for run_id in sorted(sysn_qid_cwid_pred):
        outf = '%s/%s.rr'%(outdir, run_id)
        with open(outf, 'w') as f:
            for qid in sorted(sysn_qid_cwid_pred[run_id]):
                rank = 1
                for cwid in sorted(sysn_qid_cwid_pred[run_id][qid], \
                    key=lambda d:-sysn_qid_cwid_pred[run_id][qid][d]):
                    f.write('%d Q0 %s %d %.6e %s\n'%(qid, cwid, rank, \
                        sysn_qid_cwid_pred[run_id][qid][cwid], 'rr_%s'%(run_id)))
                    rank += 1

def read_eval_res(run_ndcgerr_dir):
    rn_qid_ndcgerr=dict()
    for run_ndcgerr_f in os.listdir(run_ndcgerr_dir):
        with open(os.path.join(run_ndcgerr_dir, run_ndcgerr_f)) as f:
            
            for line in f:
                if line.startswith('runid'):
                    continue
                cols = line.split(',')
                if cols[1] != 'amean':
                    qid = int(cols[1])
                else:
                    qid = 0
                r_name, ndcg, err = cols[0], float(cols[2]), float(cols[3])
                if r_name not in rn_qid_ndcgerr:
                    rn_qid_ndcgerr[r_name] = dict()
                rn_qid_ndcgerr[r_name][qid] = ( ndcg, err )
    return rn_qid_ndcgerr

def get_rank(rr_trecrun_ndcgerr, trecrun_ndcgerr):
    def _get_rank_of_run(trecrun_scores, ind):
        sorted_runs = sorted(trecrun_scores, key=lambda rn: trecrun_scores[rn][0][ind], reverse=True)
        run_rank = dict(zip(sorted_runs, range(1, len(sorted_runs)+1)))
        return run_rank
    orig_rr_ndcg_rank, orig_rr_err_rank = dict(), dict()
    ndcg_run_rank = _get_rank_of_run(trecrun_ndcgerr, 0)
    err_run_rank = _get_rank_of_run(trecrun_ndcgerr, 1)
    for runname in trecrun_ndcgerr:
        rr_ndcg, rr_err = rr_trecrun_ndcgerr['rr_{0}'.format(runname)][0] 
        original_trecrun_ndcgerr = copy.deepcopy(trecrun_ndcgerr)
        original_trecrun_ndcgerr[runname][0] = (rr_ndcg, rr_err)
        rr_ndcg_rank = _get_rank_of_run(original_trecrun_ndcgerr, 0)[runname]
        rr_err_rank = _get_rank_of_run(original_trecrun_ndcgerr, 1)[runname]
        o_ndcg_rank, o_err_rank = ndcg_run_rank[runname], err_run_rank[runname]
        orig_ndcgs = [trecrun_ndcgerr[runname][qid][0] for qid in trecrun_ndcgerr[runname] if qid != 0]
        orig_errs = [trecrun_ndcgerr[runname][qid][1] for qid in trecrun_ndcgerr[runname] if qid != 0]
        rr_ndcgs = [rr_trecrun_ndcgerr['rr_{0}'.format(runname)][qid][0] \
                for qid in rr_trecrun_ndcgerr['rr_{0}'.format(runname)] if qid != 0]
        rr_errs = [rr_trecrun_ndcgerr['rr_{0}'.format(runname)][qid][1] \
                for qid in rr_trecrun_ndcgerr['rr_{0}'.format(runname)] if qid != 0]
        orig_rr_ndcg_rank[runname] = (o_ndcg_rank, trecrun_ndcgerr[runname][0][0],  list(orig_ndcgs), \
                rr_ndcg_rank, rr_ndcg, list(orig_errs))
        orig_rr_err_rank[runname] = (o_err_rank, trecrun_ndcgerr[runname][0][1], list(orig_errs), \
                rr_err_rank, rr_err, list(rr_errs))
    return orig_rr_ndcg_rank, orig_rr_err_rank


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
            output_ql='%s/train_%s/%s/evaluations/rerank-ql/%s_v-%s_t-%s/%s'%(p['outdir'], train_years,\
                    p['expname'], '-'.join(train_years.split('_')), val_year[2:], test_year[2:], raw_expid)
            output_rrall='%s/train_%s/%s/evaluations/rerank-all/%s_v-%s_t-%s/%s'%(p['outdir'], train_years,\
                    p['expname'], '-'.join(train_years.split('_')), val_year[2:], test_year[2:], raw_expid)
            reranked_run_dir='%s/train_%s/%s/reranking/trecrun/v%s-t%s_%s'%(p['outdir'],train_years,\
                        p['expname'],val_year, test_year,raw_expid)
            reranked_metric_dir='%s/train_%s/%s/reranking/ndcgerr/v%s-t%s_%s'%(p['outdir'],train_years,\
                        p['expname'],val_year, test_year,raw_expid)

            try:
                if not os.path.isdir(reranked_run_dir):
                    os.makedirs(reranked_run_dir)
            except OSError as e:
                print(e)
            try:
                if not os.path.isdir(reranked_metric_dir):
                    os.makedirs(reranked_metric_dir)
            except OSError as e:
                print(e)
            try:
                if not os.path.isdir(os.path.dirname(output_ql)):
                    os.makedirs(os.path.dirname(output_ql))
            except OSError as e:
                print(e)
            try:
                if not os.path.isdir(os.path.dirname(output_rrall)):
                    os.makedirs(os.path.dirname(output_rrall))
            except OSError as e:
                print(e)
            _log.info('evaluate {0} on {1} based on val {2} \
                    over docpairs benchmark and output to {3}, {4}'.format(expid, test_year, val_year, output_ql, output_rrall))
            
            
            trec_run_dir='{0}/{1}'.format(trec_run_basedir, test_year)
            eval_trecrun_dir='{0}/{1}'.format(eval_trec_run_basedir, test_year)

            test_qids = year_qids[test_year]
            qrelf = get_qrelf(qrelfdir, test_year)
            qid_cwid_label = read_qrel(qrelf, test_qids, include_spam=False)
            best_pred_dir, argmax_epoch, argmax_run, argmax_ndcg, argmax_err = get_epoch_from_val(pred_dirs, val_dirs)
            # create re-rank
            qid_cwid_pred, _, pred_expid = read_run(os.path.join(best_pred_dir, argmax_run))
            sysn_qid_cwid_pred = trec_run_predscore(_log, trec_run_dir, qid_cwid_pred)
            print_rerun(sysn_qid_cwid_pred, reranked_run_dir, pred_expid, val_year, test_year)
            # eval re-rank
            for runfile in os.listdir(reranked_run_dir):
                outfile='%s/%s.ndcg_err'%(reranked_metric_dir, runfile[:-3])
                with open(outfile, 'w') as outf:
                    subprocess.Popen([perlf, qrelf, '%s/%s'%(reranked_run_dir,runfile)], stdout=outf)
                    _log.info('finished {0} train on {1}, val on {2}, test on {3}'.format(runfile, train_years, val_year, test_year))
            # read in eval and generate results
            trecrun_qid_ndcgerr = read_eval_res(eval_trecrun_dir)
            while True:
                rr_trecrun_qid_ndcgerr = read_eval_res(reranked_metric_dir)
                _log.error('mismatched #run {0} != {1}'.format(len(trecrun_qid_ndcgerr), len(rr_trecrun_qid_ndcgerr)))
                if len(trecrun_qid_ndcgerr) == len(rr_trecrun_qid_ndcgerr):
                    break
                # latency for subprocess.Popen
                time.sleep(2)
            # orig_rank, orig_score, qidscores, rr_rank, rr_score, qidscores
            orig_rr_ndcg_rank, orig_rr_err_rank = get_rank(rr_trecrun_qid_ndcgerr, trecrun_qid_ndcgerr)

            if test_year in ['wt09', 'wt10', 'wt11', 'wt12', 'wt13', 'wt14']:
                # query likelihood benchmark
                cols=['QL-Variants', 'Measures', 'TREC', 'Trec-Rank', 'Rerank', 'Rerank-Rank', 'Comparison', 'p-value']
                tabledict=dict()
                measure_ind = {'ERR':1, 'nDCG':0}
                for j, col in enumerate(cols):
                    tabledict[col] = list()
                    for method in ['cwindri']:
                        for measure in ['ERR', 'nDCG']:
                            if j == 0:
                                tabledict[col].append(method)
                            elif j == 1:
                                tabledict[col].append(measure)
                            # original trec score
                            elif j == 2:
                                if measure == 'ERR':
                                    tabledict[col].append('%.3f'%orig_rr_err_rank[method][1])
                                elif measure == 'nDCG':
                                    tabledict[col].append('%.3f'%orig_rr_ndcg_rank[method][1])
                            # original trec rank
                            elif j == 3:
                                if measure == 'ERR':
                                    tabledict[col].append(orig_rr_err_rank[method][0])
                                elif measure == 'nDCG':
                                    tabledict[col].append(orig_rr_ndcg_rank[method][0])
                            # reranked score
                            elif j == 4:
                                if measure == 'ERR':
                                    tabledict[col].append('%.3f'%orig_rr_err_rank[method][4])
                                elif measure == 'nDCG':
                                    tabledict[col].append('%.3f'%orig_rr_ndcg_rank[method][4])
                            # reranked rank
                            elif j == 5:
                                if measure == 'ERR':
                                    tabledict[col].append(orig_rr_err_rank[method][3])
                                elif measure == 'nDCG':
                                    tabledict[col].append(orig_rr_ndcg_rank[method][3])
                            # comparison: (r-t)/t * 100 %
                            elif j == 6:
                                if measure == 'ERR':
                                    comp = (orig_rr_err_rank[method][4] - orig_rr_err_rank[method][1]) / orig_rr_err_rank[method][1]
                                    tabledict[col].append('%.0f%%'%(comp*100))
                                elif measure == 'nDCG':
                                    comp = (orig_rr_ndcg_rank[method][4] - orig_rr_ndcg_rank[method][1]) / orig_rr_ndcg_rank[method][1]
                                    tabledict[col].append('%.0f%%'%(comp*100))
                            # comparison: p-value
                            elif j == 7:
                                if measure == 'ERR':
                                    _, p_err_diff = ttest_rel(orig_rr_err_rank[method][2], orig_rr_err_rank[method][5])
                                    tabledict[col].append('%.3f'%(p_err_diff))
                                elif measure == 'nDCG':
                                    _, p_ndcg_diff = ttest_rel(orig_rr_ndcg_rank[method][2], orig_rr_ndcg_rank[method][5])
                                    tabledict[col].append('%.3f'%(p_ndcg_diff))

                dftable = pd.DataFrame(tabledict, columns=cols, index=None)
                _log.info('\n' + dftable.to_string())
                dftable.to_csv(output_ql + '.csv', float_format='%.3f', header=True, index=False, sep=',', mode='w')
                _log.info('finished ql benchmark {0} {1} {2} {3}'.format(expid, train_years, val_year, test_year))

            # re-rank all benchmark
            def comparison(orig_rr_rank):
                count = 0
                percents = list()
                for r in orig_rr_rank:
                    orig_rank, orig_score, orig_scores, rr_rank, rr_score, rr_scores = orig_rr_rank[r]
                    if rr_rank < orig_rank:
                        count += 1
                    # compute micro avg
                    qid_chg = (rr_score - orig_score) / orig_score
                    percents.append(qid_chg)
                return count, np.mean(percents), np.median(percents)

            cols=['Measures', '#Total Runs', '#Improved', 'Avg', 'Median']
            orig_rr_ranks = [orig_rr_ndcg_rank, orig_rr_err_rank]
            tabledict=list()
            for i, measure in enumerate(['nDCG', 'ERR']):
                tabledict.append(dict())
                count, avg_chg, median_chg = comparison(orig_rr_ranks[i])
                for j, col in enumerate(cols):
                    if j == 0:
                        tabledict[i][col] = measure
                    elif j == 1:
                        tabledict[i][col] = len(orig_rr_ranks[i])
                    elif j == 2:
                        tabledict[i][col] = count
                    elif j == 3:
                        tabledict[i][col] = '%.0f%%'%(avg_chg*100)
                    elif j == 4:
                        tabledict[i][col] = '%.0f%%'%(median_chg*100)

            dftable = pd.DataFrame(tabledict, columns=cols, index=None)
            _log.info('\n' + dftable.to_string())
            dftable.to_csv(output_rrall + '.csv', float_format='%.3f', header=True, index=False, sep=',', mode='w')
            _log.info('finished rerank all benchmark {0} {1} {2} {3}'.format(expid, train_years, val_year, test_year))

            # rank vs percentage of change
            def rank_improve(orig_rr_rank):
                oscore_percent = list()
                for r in orig_rr_rank:
                    orig_rank, orig_score, orig_scores, rr_rank, rr_score, rr_scores = orig_rr_rank[r]
                    percent = (rr_score - orig_score) / orig_score
                    oscore_percent.append((orig_score, percent))
                return [p for s, p in sorted(oscore_percent, key=lambda s_p:s_p[0], reverse=True)]
            def plot_curve(ranks, ndcg_ps, err_ps, outfilename):
                fig, ax = plt.subplots()
                rects1 = ax.scatter(ranks, ndcg_ps, s=25, c='b', marker="^", lw=0)
                rects2 = ax.scatter(ranks, err_ps, s=25, c='r', marker="o", lw=0)
                vals = ax.get_yticks()
                ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
                ax.set_xlabel('Rank of runs from TREC sorted by corresponding measures')
                ax.set_ylabel('Relative improvement based on Err/nDCG')
                ax.legend((rects1, rects2), ('Improvements based on nDCG', 'Improvements based on ERR'))
                plt.grid(b=False, linestyle='--')
                fig.savefig(outfilename + '.pdf', format='pdf')
                plt.close()


            ndcg_ps = rank_improve(orig_rr_ndcg_rank)
            err_ps = rank_improve(orig_rr_err_rank)
            ranks = range(1, len(ndcg_ps)+1)
            plot_curve(ranks, ndcg_ps, err_ps, output_rrall)



