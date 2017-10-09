import numpy as np
from collections import Counter
import xml.etree.ElementTree as ET
import os, getpass
import logging

def read_qrel(qrelfile, qids, include_spam):
    qid_cwid_label=dict()
    with open(qrelfile) as f:
        for line in f:
            cols = line.split()
            qid, cwid, label = int(cols[0]), cols[2], int(cols[3])
            if qid not in qids:
                continue
            if not include_spam:
                if label < 0:
                    continue
            if qid not in qid_cwid_label:
                qid_cwid_label[qid]=dict()
            qid_cwid_label[qid][cwid]=label
    return qid_cwid_label

def string_chunker(str2split, stopwords=set()):
    terms = re.split(' |\/|,|!|-|:|\'', str2split)
    terms = [t for t in [t.strip(' \t\n\r()?\.\"').lower() for t in terms if t != '\n' and t != 's'] \
            if len(t)>0]                        
    terms = [t for t in terms  if t not in stopwords or len(stopwords)==0]
    return terms

# read in query
def readinquery(sc, queryfile):
    qidTopic = dict()
    qidDesc = dict()
    query_xml = sc.wholeTextFiles(queryfile).map(lambda p_q: p_q[1]).collect()
    root = ET.fromstring(query_xml[0])
    for query in root.findall('topic'):
        qid = int(query.attrib['number'])
        for attr in query:
            if attr.tag == 'query':
                qidTopic[qid]=attr.text
            elif attr.tag == 'description':
                qidDesc[qid]=attr.text
    return qidTopic, qidDesc

def config_logger(loggername, lognamehead, currentdir, detail_outdir, expid, train_year, val_year='', test_year=''):
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    logdir = "{0}/log/".format(currentdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if lognamehead == 'train':
        fhcwd = logging.FileHandler("{0}/log/{1}_{2}_on_{3}.log".format(currentdir, lognamehead, expid, train_year))
        fhmodel = logging.FileHandler("{0}/outs/{1}_{2}_on_{3}.out".format(detail_outdir, lognamehead, expid, train_year))
    elif lognamehead == 'pred':
        fhcwd = logging.FileHandler("{0}/log/{1}_{2}_on_{3}_p{4}.log".format(os.getcwd(), lognamehead, expid, train_year, test_year))
        fhmodel = logging.FileHandler("{0}/outs/{1}_{2}_on_{3}_p{4}.out".format(detail_outdir, lognamehead, expid, train_year, test_year))
    elif lognamehead.startswith('eval'):
        fhcwd = logging.FileHandler("{0}/log/{1}_{2}_on_{3}_v{4}_t{5}.log".format\
                (os.getcwd(), lognamehead, expid, train_year, val_year, test_year))
        fhmodel = logging.FileHandler("{0}/outs/{1}_{2}_on_{3}_v{4}_t{5}.out".format\
                (detail_outdir, lognamehead, expid, train_year, val_year, test_year))
    ch = logging.StreamHandler()
    logFormatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] [%(levelname)-5.5s]  %(message)s")
    fhcwd.setFormatter(logFormatter)
    fhmodel.setFormatter(logFormatter)
    ch.setFormatter(logFormatter)
    logger.addHandler(fhcwd)
    logger.addHandler(fhmodel)
    logger.addHandler(ch)
    return logger

class SoftFailure(Exception):
    pass

