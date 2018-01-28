import os
import sys
import tempfile
import subprocess
import pickle
import time
import importlib
import sacred
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 10})
import matplotlib.pyplot as plt
from keras.utils import plot_model
import keras.backend as K
import utils.select_doc_pos
from sacred.utils import apply_backspaces_and_linefeeds
from utils.utils import load_test_data, DumpWeight, dump_modelplot, pred_label, trunc_dir
from utils.config import treceval, perlf, rawdoc_mat_dir, file2name, default_params, qrelfdir
from utils.year_2_qids import get_train_qids, get_qrelf
from utils.common_utils import read_qrel, SoftFailure
from utils.ngram_nfilter import get_ngram_nfilter

K.get_session()
ex = sacred.Experiment('predict')
ex.path = 'predict'
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('CUDA_VISIBLE_DEVICES')
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('USER')
ex.captured_out_filter = apply_backspaces_and_linefeeds

from utils.config import default_params
default_params = ex.config(default_params)


def plot_curve(epoch_err_ndcg_loss, outdir, plot_id, p):
    epoches, errs, ndcgs, maps, losses = zip(*epoch_err_ndcg_loss)
    losses = [loss/10000.0 for loss in losses]
    fig, ax = plt.subplots()
    rects1 = ax.plot(epoches, ndcgs, 'b--')
    rects2 = ax.plot(epoches, maps, color='r')
    rects3 = ax.plot(epoches, errs, 'g.')
    axt = ax.twinx()
    rects0 = axt.plot(epoches, losses, 'k:')
    axt.set_ylabel('Training Loss')
    axt.tick_params('y')
    ax.set_xlabel('Epoches')
    ax.set_ylabel('nDCG/MAP/Err')
    ax.set_title('Train %s Test %s'%(p['train_years'], p['test_year'])+\
            'Loss:%d %.3f'%(epoches[np.argmin(losses)], losses[np.argmin(losses)]) +\
            ' Err:%d %.3f'%(epoches[np.argmax(errs)], errs[np.argmax(errs)]) +\
            ' MAP:%d %.3f'%(epoches[np.argmax(maps)], maps[np.argmax(maps)]) +\
            ' nDCG:%d %.3f'%(epoches[np.argmax(ndcgs)], ndcgs[np.argmax(ndcgs)]))
    ax.legend((rects0[0], rects1[0], rects2[0], rects3[0]), ('Train Loss', 'Prediction nDCG', \
            'Prediction MAP','Prediction Err'), loc='center right')
    fig.savefig(os.path.join(outdir, plot_id + '.pdf'))
    plt.close()


def eval_run(_log, qid_cwid_pred, expid, perlf, treceval, tmp_dir, k, qrelf):
    with tempfile.NamedTemporaryFile(mode='w', delete=True, dir=tmp_dir) as tmpf,\
            open(os.path.join(tmp_dir, 'tmperr.f'),'a+') as errf:
        for qid in sorted(qid_cwid_pred):
            rank = 1
            for cwid in sorted(qid_cwid_pred[qid], key=lambda d:-qid_cwid_pred[qid][d]):
                tmpf.write('%d Q0 %s %d %.10e %s\n'%(qid, cwid, rank, qid_cwid_pred[qid][cwid], expid))
                rank += 1
        tmpf.flush()
        run2eval = tmpf.name
        try:
            val_res = subprocess.check_output([perlf, '-k','%d'%k, qrelf, run2eval], stderr=errf).decode('utf-8')
            map_res = subprocess.check_output([treceval, '-m','map', qrelf, run2eval], stderr=errf).decode('utf-8')
        except subprocess.CalledProcessError as e:
            _log.error(e)
            exit(1)
    amean_line = val_res.splitlines()[-1]
    mapval = map_res.split()[-1]
    if 'amean' not in amean_line:
        _log.error('Error in validation: %s'%amean_line)
    cols = amean_line.split(',')
    ndcg20, err20, mapv = float(cols[-2]), float(cols[-1]), float(mapval)
    return ndcg20, err20, mapv


def print_run(qid_cwid_pred, outdir, outfname, run_id):
    with open(os.path.join(outdir, outfname), 'w') as f:
        for qid in sorted(qid_cwid_pred):
            rank = 1
            for cwid in sorted(qid_cwid_pred[qid], key=lambda d:-qid_cwid_pred[qid][d]):
                f.write('%d Q0 %s %d %.10e %s\n'%(qid, cwid, rank, qid_cwid_pred[qid][cwid], run_id))
                rank += 1


@ex.automain
def pred(_log, _config):
    p = _config

    modelname = file2name[p['modelfn']]
    mod_model = importlib.import_module('models.%s' % p['modelfn'])
    model_cls = getattr(mod_model, modelname)
    model_params = {k: v for k, v in p.items() if k in model_cls.params or k == 'modelfn'}
    model = model_cls(model_params, rnd_seed=p['seed'])
    expid = model.params_to_string(model_params)

    outdir_plot=trunc_dir('%s/train_%s/%s/predict_per_epoch/test_%s' % (p['parentdir'], p['train_years'],
                                                              p['expname'], p['test_year']))
    outdir_run=trunc_dir('%s/%s'%(outdir_plot, expid))
    tmp_dir=trunc_dir(os.path.join(outdir_run,'tmp'))
    weight_dir=trunc_dir('%s/train_%s/%s/model_weight/%s' % (p['parentdir'], p['train_years'],p['expname'], expid))
    detail_outdir=trunc_dir('%s/train_%s/%s/model_detail/' % (p['parentdir'], p['train_years'], p['expname']))

    assert os.path.isdir(weight_dir), "weight_dir " + weight_dir + " does not exist. Make sure you trained the model."
    assert os.path.isdir(detail_dir), "detail_dir " + detail_dir + " does not exist. Make sure you trained the model."

    if len(os.listdir(weight_dir)) < 1:
        raise SoftFailure('weight dir empty')

    try:
        if not os.path.isdir(outdir_run):
            os.makedirs(outdir_run)
            os.makedirs(tmp_dir)
    except OSError:
        pass
    _log.info('Processing {0}'.format(outdir_run))
    ###################
    label2tlabel={4:2,3:2,2:2,1:1,0:0,-2:0}
    topk4eval=20
    NGRAM_NFILTER, N_GRAMS = get_ngram_nfilter(p['winlen'], p['qproximity'], p['maxqlen'], p['xfilters'])

    _log.info('process {0} and output to {1}'.format(weight_dir, outdir_run))
    _log.info('{0} {1} {2} {3} {4}'.format(p['distill'], 'NGRAM_NFILTER', NGRAM_NFILTER, 'N_GRAMS', N_GRAMS))

    # prepare train data
    qids = get_train_qids(p['test_year'])
    qrelf = get_qrelf(qrelfdir, p['test_year'])
    qid_cwid_label = read_qrel(qrelf, qids, include_spam=False)
    test_qids =[qid for qid in qids if qid in qid_cwid_label]
    _log.info('%s test_num %d '%(p['test_year'], len(test_qids)))

    f_ndcg=dict()
    f_epochs = set()
    # sort weights by time and only use the first weights for each epoch
    # (in case there are duplicate weights from a failed/re-run train)
    for f in sorted(os.listdir(weight_dir),
                    key=lambda x: os.path.getctime(os.path.join(weight_dir, x))):
        if f.split('.')[-1] != 'h5':
            continue
        cols = f.split('.')[0].split('_')
        if len(cols) == 4:
            nb_epoch, loss, n_batch, n_samples = int(cols[0]), int(cols[1]), int(cols[2]), int(cols[3])
            if nb_epoch <= p['epochs'] and nb_epoch not in f_epochs:
                f_epochs.add(nb_epoch)
                f_ndcg[f]=(nb_epoch, loss, n_batch, n_samples)


    finished_epochs = {}
    for fn in sorted(os.listdir(outdir_run),
                     key=lambda x: os.path.getctime(os.path.join(outdir_run, x))):
        if fn.endswith(".run"):
            fields = fn[:-4].split("_") # trim .run
            assert len(fields) == 5

            epoch, loss = int(fields[0]), int(fields[4])
            ndcg, mapv, err = float(fields[1]), float(fields[2]), float(fields[3])

            #assert epoch not in finished_epochs
            if epoch in finished_epochs:
                _log.error("TODO two weights exist for same epoch")
            finished_epochs[epoch] = (epoch, err, ndcg, mapv, loss)

    _log.info('skipping finished epochs: {0}'.format(finished_epochs))

    def model_pred(NGRAM_NFILTER, weight_file, test_data, test_docids, test_qids):
        dump_modelplot(model.build(), detail_outdir + 'predplot_' + expid)
        model_predict = model.build_from_dump(weight_file)
        qid_cwid_pred = pred_label(model_predict, test_data, test_docids, test_qids)
        return qid_cwid_pred

    test_doc_vec, test_docids, test_qids=load_test_data(qids, rawdoc_mat_dir, qid_cwid_label, N_GRAMS, p)
    epoch_err_ndcg_loss=list()
    _log.info('start {0} {1} {2}'.format(expid, p['train_years'], p['test_year']))
    for f in sorted(f_ndcg, key=lambda x:f_ndcg[x][0]):
        nb_epoch, loss, n_batch, n_samples = f_ndcg[f]
        if nb_epoch in finished_epochs:
            epoch_err_ndcg_loss.append(finished_epochs[nb_epoch])
            continue
        weight_file = os.path.join(weight_dir, f)
        qid_cwid_pred = model_pred(NGRAM_NFILTER, weight_file, test_doc_vec, test_docids, test_qids)
        ndcg20, err20, mapv = eval_run(_log, qid_cwid_pred, expid, perlf, treceval, tmp_dir, topk4eval, qrelf)
        loss = int(loss)
        out_name = '%d_%0.4f_%0.4f_%0.4f_%d.run' % (nb_epoch, ndcg20, mapv, err20, loss)
        epoch_err_ndcg_loss.append((nb_epoch, err20, ndcg20, mapv, loss))
        print_run(qid_cwid_pred, outdir_run, out_name, expid)
        _log.info('finished {0}'.format(f))
    _log.info('finish {0} {1} {2}'.format(expid, p['train_years'], p['test_year']))

    plot_curve(epoch_err_ndcg_loss, outdir_plot, expid, p)

    if max(f_epochs) < p['epochs'] - 3:
        raise SoftFailure("prediction finished, but not all epochs are available yet. last epoch found: %s" % max(f_epochs))
