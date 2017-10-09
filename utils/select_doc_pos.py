import numpy as np
from itertools import groupby
from operator import itemgetter
def _moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def select_pos_firstk(sim_mat, dim_sim=112, win_len=3):
    return list(range(dim_sim))

def select_pos_absolutemax(sim_mat, dim_sim=112, win_len=3):
    max_v = sim_mat.max(axis=0)
    inds = np.argsort(-max_v) 
    return sorted(inds[:dim_sim])


def select_pos_maxslide(sim_mat, dim_sim=112, win_len=3):
    max_v = sim_mat.max(axis=0)
    avg_max = _moving_average(max_v, win_len) 
    inds = np.argsort(-avg_max)
    selected_pos=set()
    for i in inds:
        selected_pos.update(range(i, i+win_len))
        if len(selected_pos) >= dim_sim:
            break
    return sorted(list(selected_pos))[:dim_sim]

def select_pos_strides(sim_mat, dim_sim=112, win_len=3):
    max_v = sim_mat.max(axis=0)
    avg_max = _moving_average(max_v, win_len) 
    inds = np.argsort(-avg_max)
    selected_pos=list()
    for i in inds:
        for j in range(i, i+win_len):
            if i+win_len < sim_mat.shape[1]:
                selected_pos.append(j)
        if len(selected_pos) >= dim_sim:
            break
    return selected_pos[:dim_sim]

def select_pos_maxslidesep(sim_mat, dim_sim=112, win_len=3):
    selected_pos = select_pos_maxslide(sim_mat, dim_sim=dim_sim, win_len=win_len)
    sep_pos = list()
    for k, g in groupby(enumerate(selected_pos), lambda ind_x: ind_x[0]-ind_x[1]):
        for ind, v in g:
            sep_pos.append(v)
        sep_pos.append(-1)
        if len(sep_pos) >= dim_sim:
            break
    return sep_pos[:dim_sim]


def select_pos_maxslideold(sim_mat, dim_sim=112, win_len=3):
    max_v = sim_mat.max(axis=0)
    avg_max = _moving_average(max_v, win_len)
    inds = np.argsort(-avg_max)
    selected_pos=list()
    for w in range(win_len):
        selected_pos += (inds+w).tolist()
    return sorted(selected_pos[:dim_sim])
