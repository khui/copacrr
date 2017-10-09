import numpy as np
from collections import Counter
import itertools
import xml.etree.ElementTree as ET
import os
import numpy as np
from collections import Counter

import math
import sys
sys.path.append('/home/khui/workspace/python/docrep/lib')
from common_utils import *


def queryterm_docterm_similarity(qid, qid_topic_terms, qid_desc_terms, qid_cwid_label, \
        cwid_docs, w2v_model, include_desc=False):
    qterms_dterms_simi = dict()
    cwids = set(qid_cwid_label[qid].keys()) & set(cwid_docs.keys())
    query_terms = qid_topic_terms[qid]
    if include_desc:
        query_terms = query_terms + qid_desc_terms[qid]
    doc_terms = set([t for cwid in cwids for t in cwid_docs[cwid]])
    for qt in set(query_terms):
        qterms_dterms_simi[qt]=dict()
        for dt in doc_terms:
            if qt in w2v_model and dt in w2v_model:       
                similarity = w2v_model.similarity(qt, dt)
                qterms_dterms_simi[qt][dt] = similarity
    return qterms_dterms_simi

def query_doc_simialrity(qid, qid_topic_terms, qid_desc_terms, \
        qid_cwid_label, qterms_dterms_simi, cwid_docs, include_desc=False):
    query_terms = qid_topic_terms[qid]
    if include_desc:
        query_terms = query_terms + qid_desc_terms[qid]
    query_terms = set(query_terms)
    qtCwidSimi = dict()
    for cwid in qid_cwid_label[qid]:
        if cwid not in cwid_docs:
            continue
        doc_terms = cwid_docs[cwid]
        for qt in query_terms:
            if qt not in qtCwidSimi:
                qtCwidSimi[qt] = dict()
            qtCwidSimi[qt][cwid]=list()
            for dt in doc_terms:
                if qt in qterms_dterms_simi:
                    if dt in qterms_dterms_simi[qt]:
                        similarity = qterms_dterms_simi[qt][dt]
                        qtCwidSimi[qt][cwid].append(similarity)
    return qtCwidSimi

def convert2simsquare(topic_terms, desc_terms, qterm_docsims, cwid_label, include_desc=False, SIM_DIM=30):
    pos_cwid_simmaxtrix = dict()
    neg_cwid_simmaxtrix = dict()
    qts = list(topic_terms)
    if include_desc:
        qts = qts + list(desc_terms)
    for qt in qts:
        for cwid in qterm_docsims[qt]:
            if len(qterm_docsims[qt][cwid]) < SIM_DIM: 
                qterm_docsims[qt][cwid] = [0] * SIM_DIM
    cwids = set(itertools.chain.from_iterable([qterm_docsims[qt].keys() for qt in qts]))
    if len(qts) == 0:
        return dict(), dict(), qts
    for cwid in cwids:
        label = cwid_label[cwid]
        to_include = True
        if label > 0:
            pos_cwid_simmaxtrix[cwid]=list()
            for qt in qts:    
                pos_cwid_simmaxtrix[cwid].append(list(qterm_docsims[qt][cwid]))
            pos_cwid_simmaxtrix[cwid] = np.array(pos_cwid_simmaxtrix[cwid])
        elif label == 0:
            neg_cwid_simmaxtrix[cwid]=list()
            for qt in qts:
                neg_cwid_simmaxtrix[cwid].append(list(qterm_docsims[qt][cwid]))
            neg_cwid_simmaxtrix[cwid] = np.array(neg_cwid_simmaxtrix[cwid])
    return pos_cwid_simmaxtrix, neg_cwid_simmaxtrix, qts

def convert2testdata(qid, pos_cwid_simmaxtrix, neg_cwid_simmaxtrix, query_terms, \
                qid_cwid_label, qid_term_idf, SIM_DIM=30, N_QUERY_TERM=10): 
    qtermnum = len(query_terms)
    term_idf = qid_term_idf[qid]
    doc_cwid_simmatrix = dict(pos_cwid_simmaxtrix.items(), **neg_cwid_simmaxtrix)
    docnum = len(doc_cwid_simmatrix)
    doc_vec = np.zeros((docnum, N_QUERY_TERM, SIM_DIM))
    qidfvecs = np.zeros((docnum, N_QUERY_TERM, 1))
    doc_vec[:,:qtermnum,:] = np.array([doc_cwid_simmatrix[cwid]  \
              for cwid in sorted(doc_cwid_simmatrix)])
    qidfvecs[:,:qtermnum,:] = \
            np.array([term_idf[qt] for qt in query_terms]).reshape(1,qtermnum).repeat(docnum, axis=0).reshape((docnum,qtermnum,1))
    cwids = [cwid for cwid in sorted(doc_cwid_simmatrix)]
    return doc_vec, qidfvecs, cwids


def read_query_idf(sc, qterm_df_idf):
    term_df_idf_09 =\
    sc.textFile(qterm_df_idf + 'cw09/part*').map(lambda line: line.split(' '))\
    .map(lambda cols: (cols[0], (int(cols[1]), float(cols[2])))).collect()

    term_df_idf_12 =\
    sc.textFile(qterm_df_idf + 'cw12/part*').map(lambda line: line.split(' '))\
    .map(lambda cols: (cols[0], (int(cols[1]), float(cols[2])))).collect()

    term_df_idf_09 = dict(term_df_idf_09)
    term_df_idf_12 = dict(term_df_idf_12)
    return term_df_idf_09, term_df_idf_12

#def write_test_data(outdir, qid_topic_terms, qid_desc_terms,qid_cwid_label, qid_term_idf,\
#                    cwid_docs, word2vec_model):
#    qids = range(101,301)
#    similarity_filter = Sim_Filter_Slide(win_len=WIN_LEN, doc_dim=SIM_DIM)
#    test_doc_vec, test_query_idfs, test_docids, test_qids =\
#            load_test_data(qids, qid_topic_terms, qid_desc_terms,\
#            qid_cwid_label, qid_term_idf, \
#            cwid_docs, word2vec_model, include_desc=True, \
#            similarity_filter=similarity_filter, SIM_DIM=SIM_DIM, N_QUERY_TERM=MAX_QUERY_LENGTH)


def dump_simmat_perq(outdir, qids, qid_topic_terms, qid_desc_terms,\
                qid_cwid_label, qid_term_idf, \
              cwid_docs, w2v_model, include_desc=True):
    testdocs, qidfs, docids, testqids = list(), list(), list(), list()
    for qid in qids:
        if qid not in qid_cwid_label:
            print('Error %d not in qrel'%qid)
            continue
        cwid_label = qid_cwid_label[qid]
        qterms_dterms_simi = \
        queryterm_docterm_similarity(qid, qid_topic_terms, qid_desc_terms,\
                                        qid_cwid_label, \
                                     cwid_docs, w2v_model, include_desc=include_desc)
        qterm_docsims = \
        query_doc_simialrity(qid, qid_topic_terms, qid_desc_terms, \
        qid_cwid_label, qterms_dterms_simi, cwid_docs, include_desc=include_desc)
        
        term_idf = qid_term_idf[qid]
        for qt in qterm_docsims:
            outsubdir = outdir + '/%d/%s'%(qid, qt)
            if not os.path.isdir(outsubdir):
                os.makedirs(outsubdir)
            for cwid in qterm_docsims[qt]:
                out_file = outsubdir + '/' + cwid
                np.save(out_file + '.npy', np.array(qterm_docsims[qt][cwid]))
        with open(outdir + '/%d/qt_idf.txt'%qid, 'w') as outf:
            for qt in sorted(term_idf):
                outf.write('%s %.8e\n'%(qt, term_idf[qt]))

def dump_query_idf(qids, doc_mat_dir, qid_topic_terms, qid_desc_terms):
    qid_qterms_idf = dict()
    qid_topic_arr = dict()
    qid_desc_arr = dict()
    outdir_topic = doc_mat_dir + '/query_idf/topic_term_idf'
    outdir_desc = doc_mat_dir + '/query_idf/desc_term_idf'
    if not os.path.isdir(outdir_topic):
        os.makedirs(outdir_topic)
        os.makedirs(outdir_desc)
    for qid in qids:
        topic_terms, desc_terms = qid_topic_terms[qid], qid_desc_terms[qid]
        qid_qterms_idf[qid]=dict()
        query_idf_f = doc_mat_dir + '/per_query_term/%d/qt_idf.txt'%qid
        if not os.path.isfile(query_idf_f):
            print('Error %s not exist'%query_idf_f)
            continue
        with open(query_idf_f) as f:
            for line in f:
                cols = line.split(' ')
                qt, idf = cols[0], float(cols[1].rstrip('\n'))
                qid_qterms_idf[qid][qt]=idf
        qid_topic_arr[qid] = np.array([qid_qterms_idf[qid][qt] for qt in topic_terms])
        qid_desc_arr[qid] = np.array([qid_qterms_idf[qid][qt] for qt in desc_terms])
        np.save(outdir_topic + '/%d.npy'%qid, qid_topic_arr[qid])
        np.save(outdir_desc + '/%d.npy'%qid, qid_desc_arr[qid])
    return qid_topic_arr, qid_desc_arr


def dump_docmat(qids, qid_topic_terms, qid_desc_terms, qid_cwid_label, base_dir):
    sim_mat_dir = base_dir + '/per_query_term'
    dump_query_idf(qids, base_dir, qid_topic_terms, qid_desc_terms)
    for qid in qids:
        if not os.path.isdir(sim_mat_dir + '/%d'%qid):
            print('ERROR: %d does not exist'%qid)
            continue
        topic_out_dir = base_dir + '/topic_doc_mat/%d'%qid
        desc_out_dir = base_dir + '/desc_doc_mat/%d'%qid
        if not os.path.isdir(topic_out_dir):
            os.makedirs(topic_out_dir)
            os.makedirs(desc_out_dir)
        if qid not in qid_cwid_label:
            print('Error %d not in qrel'%qid)
            continue
        topic_terms = qid_topic_terms[qid]
        desc_terms = qid_desc_terms[qid]
        for cwid in qid_cwid_label[qid]:
            vecs = dict()
            vec_lends = list()
            for qt in set(topic_terms+desc_terms):
                cwidf = sim_mat_dir + '/%d/%s/%s.npy'%(qid,qt,cwid)
                if os.path.isfile(cwidf):
                    vecs[qt] = np.load(cwidf)
                    vec_lends.append(vecs[qt].shape[0])
            if len(vec_lends) == 0:
                print('ERROR: %d, %s does not include any query terms!'%(qid, cwid))
                continue
            doc_len = max(vec_lends)
            doc_mat = list()

            for qt in topic_terms:
                if qt not in vecs:
                    doc_mat.append(np.zeros(doc_len))
                elif vecs[qt].shape[0] < doc_len:
                    doc_mat.append(np.pad(vecs[qt],(0, doc_len-vecs[qt].shape[0]),mode='constant', constant_values=0))
                else:
                    doc_mat.append(vecs[qt])
            topic_doc_mat = np.array(doc_mat)
            doc_mat = list()
            for qt in desc_terms:
                if qt not in vecs:
                    doc_mat.append(np.zeros(doc_len))
                elif vecs[qt].shape[0] < doc_len:
                    doc_mat.append(np.pad(vecs[qt],(0, doc_len-vecs[qt].shape[0]),mode='constant', constant_values=0))
                else:
                    doc_mat.append(vecs[qt])
            desc_doc_mat = np.array(doc_mat)
            np.save(topic_out_dir + '/%s.npy'%cwid, topic_doc_mat)
            np.save(desc_out_dir + '/%s.npy'%cwid, desc_doc_mat)
        print('Finished %d'%qid)

def process_trecrun(orig_trecrun_dir, year, qid_cwid_label, outdir):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    run_dir = orig_trecrun_dir + '/' + year
    qid_trecrun = dict()
    for fname in os.listdir(run_dir):
        if not os.path.isfile(run_dir + '/' + fname):
            continue
        with open(run_dir + '/' + fname) as f:
            for line in f:
                cols = line.split()
                if len(cols) != 6:
                    continue
                try:
                    qid, cwid, r, score, sysname =\
                    int(cols[0]), cols[2], cols[3], float(cols[4]), cols[5]
                    if is_number(r):
                        r = float(r)
                    else:
                        r = -1
                except:
                    print(line)
                if sysname not in qid_trecrun:
                    qid_trecrun[sysname] = dict()
                if qid not in qid_trecrun[sysname]:
                    qid_trecrun[sysname][qid] = list()
                qid_trecrun[sysname][qid].append((qid, cwid, r, score, sysname))
    for sysname in sorted(qid_trecrun):
        qid_count = dict()
        if not os.path.isdir('%s/%s'%(outdir, year)):
            os.makedirs('%s/%s'%(outdir, year))
        with open('%s/%s/%s'%(outdir, year, sysname), 'w') as outf:
            for qid in sorted(qid_trecrun[sysname]):
                if qid not in qid_count:
                    qid_count[qid]=list()
                rank = 1
                for qid, cwid, _, score, sysname in \
                    sorted(qid_trecrun[sysname][qid], key=lambda l:(-l[3], l[2])):
                    if qid not in qid_cwid_label:
                        print('no %d in qrel'%qid)
                        continue
                    if cwid in qid_cwid_label[qid]:
                        outf.write('%d Q0 %s %d %.5f %s\n'%(qid, cwid, rank, score, sysname))
                        qid_count[qid].append(cwid)
                        rank += 1
        counts = [len(qid_count[qid]) for qid in qid_count]     
        avgcount_per_query, mincount_per_query = np.mean(counts), np.min(counts)
        print('%s %s %.0f %.0f'%(year, sysname, avgcount_per_query, mincount_per_query))
