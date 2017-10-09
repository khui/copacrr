from pyspark import SparkContext, SparkConf
from gensim.models import Word2Vec
import numpy as np
import sys, getopt
sys.path.append('/home/khui/workspace/python/docrep/lib')
from dump_data_utils import *
from common_utils import *
from rank_metrics import *

cwtitle="/user/khui/data/ExtractCwdocs/cwidtitle"
cwmdoc="/user/khui/data/ExtractCwdocs/cwidmaincontent"
cwdoc="/user/khui/data/ExtractCwdocs/cwidDoccontent"
#cluew2v="/GW/D5data-2/khui/w2vpretrained/GoogleNews-clueweb1114-query/gensim-w2v-300"
qrelf="/user/khui/data/qrel/qrels.adhoc.wt*"
queryfile = '/user/khui/data/query/wtff.xml'

stopwordlist="/GW/D5data-2/khui/stopwordlist/stop429.txt"
qterm_df_idf='/user/khui/data/query/queryterm_df_idf/'
qterm_df_cw09='/user/khui/data/cwterms/term-df-cw09'
qterm_df_cw12='/user/khui/data/cwterms/term-df-cw12'
DOCFREQUENCY_CW09 = 5180307
DOCFREQUENCY_CW12 = 7496964

rnd_seed = 7541
np.random.seed(rnd_seed)

base_dir='/GW/D5data-2/khui/docrep/dump_sim_matrix/sim_mat_6y/cosine'


conf = (SparkConf()
        .setAppName("Dump Query Terms X Doc Terms")
        .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
        .set("spark.local.dir","/GW/D5data-2/khui/cw-docvector-termdf/tmp")
        .set("spark.driver.maxResultSize", "4g")
        .set("spark.kryoserializer.buffer.mb","256"))
sc = SparkContext(conf = conf)

# read in data
stopwords = readStopword(sc, stopwordlist)
qid_topic, qid_desc = readinquery(sc, queryfile)
qid_topic_terms = {qid:string_chunker(qid_topic[qid], stopwords) for qid in qid_topic}
qid_desc_terms = {qid:string_chunker(qid_desc[qid], stopwords) for qid in qid_desc}
qids=list(range(101,301))
qid_cwid_label=relgradedqrel(sc, qrelf, qids=qids, include_spam=True, parNum=32)
dump_docmat(qids, qid_topic_terms, qid_desc_terms, qid_cwid_label, base_dir)


print('INFO finished')
