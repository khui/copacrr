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
cluew2v="/GW/D5data-2/khui/w2vpretrained/GoogleNews-wt6y-qrel/gensim-w2v-300"
#"/GW/D5data-2/khui/w2vpretrained/GoogleNews-wt0914-query/gensim-w2v-300"
qrelf="/user/khui/data/qrel/qrels.adhoc.wt*"
queryfile = '/user/khui/data/query/wtff.xml'

stopwordlist="/GW/D5data-2/khui/stopwordlist/stop429.txt"
qterm_df_idf='/user/khui/data/query/queryterm_df_idf/'
qterm_df_cw09='/user/khui/data/cwterms/term-df-cw09'
qterm_df_cw12='/user/khui/data/cwterms/term-df-cw12'
DOCFREQUENCY_CW09 = 5180307
DOCFREQUENCY_CW12 = 7496964

qids=list(range(1,301))

rnd_seed = 7541
np.random.seed(rnd_seed)

outdir='/GW/D5data-2/khui/docrep/train_test_data/sim_mat_6y/cosine'

print(outdir)

conf = (SparkConf()
        .setAppName("Dump_Sim_Matrix")
        .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
        .set("spark.local.dir","/GW/D5data-2/khui/cw-docvector-termdf/tmp")
        .set("spark.driver.maxResultSize", "4g")
        .set("spark.kryoserializer.buffer.mb","256"))
sc = SparkContext(conf = conf)

# read in data
word2vec_model = Word2Vec.load(cluew2v)
stopwords = readStopword(sc, stopwordlist)
qid_cwid_label=relgradedqrel(sc, qrelf, parNum=32, qids=list(range(1,301)), include_spam=True)
qid_topic, qid_desc = readinquery(sc, queryfile)
qid_topic_terms = {qid:string_chunker(qid_topic[qid], stopwords) for qid in qid_topic}
qid_desc_terms = {qid:string_chunker(qid_desc[qid], stopwords) for qid in qid_desc}
cwiddoccontent = load_cwiddocs(sc, cwdoc)
cwid_docs = {cwid:string_chunker(doc, stopwords) for cwid, doc in cwiddoccontent.items()}
docterms = list(itertools.chain.from_iterable(cwid_docs.values()))
queryterms = list(itertools.chain.from_iterable(qid_desc_terms.values())) +\
list(itertools.chain.from_iterable(qid_topic_terms.values()))
terms, term_termid, termid_term = constructVocbulary(docterms + queryterms)
term_df_idf_09, term_df_idf_12 = read_query_idf(sc, qterm_df_idf)
qid_term_idf, qid_missrate = convert_query_idf(qids, qid_topic_terms, qid_desc_terms, term_df_idf_09, term_df_idf_12, \
                                               include_desc=True)


dump_simmat_perq(outdir, qids, qid_topic_terms, qid_desc_terms,\
                        qid_cwid_label, qid_term_idf, \
                cwid_docs, word2vec_model, include_desc=True)



print('INFO finished')
