import os, sys, getopt
import numpy as np
from random import shuffle
import gensim
from gensim.models import Word2Vec
# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 
import datetime
import xml.etree.ElementTree as ET
'''
based on the examples from https://radimrehurek.com/gensim/models/word2vec.html
'''



opts,args=getopt.getopt(sys.argv[1:],'d:q:o:g:')
for opt,arg in opts:
    if opt in ('-d','--datadir'):
        cwid_txt_dir=str(arg)
    if opt in ('-q','--trecqueryfile'):
        query_xml_file=str(arg)
    if opt in ('-o','--outdir'):
        outdir=str(arg)
    if opt in ('-g','--googlepretrain'):
        g_pretrain_bin_file=str(arg)

word_dim=300


# read in query
def readinquery(queryfile):
    qidTopic = dict()
    tree = ET.parse(queryfile)
    root = tree.getroot()
    for query in root.findall('topic'):
        qid = int(query.attrib['number'])
        for attr in query:
            if attr.tag == 'query':
                qidTopic[qid]=attr.text
    return qidTopic

def read_docno_content(cwid_txt_dir):
    for root, _, ext_fs in os.walk(cwid_txt_dir): 
        for ext_f in ext_fs:
            if ext_f.startswith("part"):
                with open(os.path.join(root, ext_f), 'rt', encoding='utf-8') as f:
                    for line in f:
                        cols = line.split()
                        if len(cols) > 1:
                            yield cols[0], ' '.join(cols[1:])

                           

qidTopic = readinquery(query_xml_file)
# each line corresponds to one document
alldocs = []  # will hold all docs in original order
alltags = []

cwids_docs = read_docno_content(cwid_txt_dir)
for line_no, line in cwids_docs:
    docid = line_no.rstrip()
    words = gensim.utils.to_unicode(line.rstrip(), errors='strict').split()
    alldocs.append(words)
    alltags.append(docid)

for qid, query in qidTopic.items():
    words = gensim.utils.to_unicode(query.rstrip(), errors='strict').split()
    alldocs.append(words)
    alltags.append(str(docid))

doc_list = alldocs[:]  # for reshuffling per pass

print('Input %d docs in total' % (len(doc_list)))

assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"



#load_word2vec_format(googlenewscorpus, binary=True)
model = Word2Vec(size=word_dim, window=10, min_count=1, workers=32)
model.build_vocab(alldocs)
# only train for unseen words, retain the existing term vector in goole pre-trained 
model.intersect_word2vec_format(g_pretrain_bin_file, binary=True, lockf=0.0)



@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def cwidvec2str(cwid, vec):
    line=list()
    line.append(cwid)
    for idx, val in enumerate(vec):
        line.append(str(idx) + ":" + '%.6f'%val)
    return ' '.join(line)


alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

print("START training at %s" % (datetime.datetime.now()))

for epoch in range(passes):
    shuffle(doc_list)  # shuffling gets best results
    duration = 'na'
    model.alpha, model.min_alpha = alpha, alpha
    with elapsed_timer() as elapsed:
        model.train(doc_list, total_examples=len(doc_list), epochs=model.iter)
        duration = '%.1f' % elapsed()
    print('INFO: completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("INFO: all passes completed with %d terms "%len(model.wv.vocab))
if not os.path.exists(outdir):
    os.makedirs(outdir)
model.save(outdir + "/gensim-w2v-" + str(word_dim))

print("INFO: finished dumping %d at %s" % (word_dim, str(datetime.datetime.now())))
