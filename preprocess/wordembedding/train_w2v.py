import os, sys, getopt
import os.path
import numpy as np
from random import shuffle
import gensim
from gensim.models import Word2Vec
# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 
import datetime
from pyspark import SparkContext, SparkConf
import xml.etree.ElementTree as ET

opts,args=getopt.getopt(sys.argv[1:],'i:d:q:o:')
for opt,arg in opts:
    if opt in ('-d','--datadir'):
        datadir=str(arg)
    if opt in ('-i','--lineno2cwid'):
        cwiddir=str(arg)
    if opt in ('-q','--qid'):
        qid=str(arg)
    if opt in ('-o','--outdir'):
        outdir=str(arg)

word_dim=300

# train based on the pretrained...
googlenewscorpus="/GW/D5data-2/khui/w2vpretrained/GoogleNews-vectors-negative300.bin"
cwdocplaintxt="/user/khui/data/ExtractCwdocs/cwidDoccontent"
outdir="/GW/D5data-2/khui/w2vpretrained/GoogleNews-wt6y-qrel"
queryfile = '/user/khui/data/query/wtff.xml'


conf = (SparkConf()
        .setAppName("w2v for clueweb")
        .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
        .set("spark.local.dir","/GW/D5data-2/khui/cw-docvector-termdf/tmp")
        .set("spark.driver.maxResultSize", "2g")
        .set("spark.kryoserializer.buffer.mb","128"))
sc = SparkContext(conf = conf)

# read in query
def readinquery(queryfile):
    qidTopic = dict()
    query_xml = sc.wholeTextFiles(queryfile).map(lambda p_q: p_q[1]).collect()
    root = ET.fromstring(query_xml[0])
    for query in root.findall('topic'):
        qid = int(query.attrib['number'])
        for attr in query:
            if attr.tag == 'query':
                qidTopic[qid]=attr.text
    return qidTopic


cwids_docs = \
sc.textFile(cwdocplaintxt + "/*/part*", 32)\
.map(lambda line: line.split(' '))\
.map(lambda cols: (cols[0], ' '.join(cols[1:])))\
.collect()

qidTopic = readinquery(queryfile)
# each line corresponds to one document
alldocs = []  # will hold all docs in original order
alltags = []

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
model.intersect_word2vec_format(googlenewscorpus, binary=True, lockf=0.0)



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
        model.train(doc_list)
        duration = '%.1f' % elapsed()
        #print("%i passes : %s %ss" % (epoch + 1, name, duration))
    print('INFO: completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("INFO: all passes completed with %d terms ", len(model.vocab))
if not os.path.exists(outdir):
    os.makedirs(outdir)
model.save(outdir + "/gensim-w2v-" + str(word_dim))

print("INFO: finished dumping %d at %s" % (word_dim, str(datetime.datetime.now())))
