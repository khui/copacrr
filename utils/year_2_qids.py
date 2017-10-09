import os

qid_year=dict(zip(list(range(1,301)), ['wt09'] * 50 + ['wt10'] * 50 + \
                ['wt11'] * 50 + ['wt12'] * 50 + ['wt13'] * 50 + ['wt14'] * 50))

year_qids={'wt09':list(range(1,51)),'wt10':list(range(51,101)),\
               'wt11':list(range(101,151)),'wt12':list(range(151,201)),\
               'wt13':list(range(201,251)), 'wt14':list(range(251,301)), 'wttest':list(range(251,266))}

def get_train_qids(year, years=['wt09', 'wt10', 'wt11', 'wt12', 'wt13', 'wt14']):
    if year.startswith('wt'):
        prefix = 'wt'
    a_qids = list()
    for y in year[len(prefix):].split('_'):
        a_qids.extend(year_qids['%s%s'%(prefix, y)])
    return a_qids

def get_qrelf(basepath, year):
    if year.startswith("nwt") or year.startswith("wt"):
        qrelf = os.path.join(basepath, 'qrels.adhoc.6y')
    else:
        print("WARNING: no qrelf exists for get_train_qids on year: %s" % year)
        qrelf = None
        
    return qrelf

