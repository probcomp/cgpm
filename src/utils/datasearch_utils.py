import numpy as np
import crosscat.utils.general_utils as gu

def compute_betabinomial_conjugate_p(hypers):
    beta_1 = hypers[0]
    beta_0 = hypers[1]
    p = beta_1 / (beta_0 + beta_1)
    return p

def flip(p, rng):
    return rng.choice([0, 1], p=[1-p, p])

def is_binary(value):
    out = False
    if value == 0 or value == 1:
        out = True
    return out

def lognormalize(lst):
    log_z = logsumexp(lst)
    if log_z == -float("inf"):
        return [-float("inf") for x in lst]
    else:
        return [x - logsumexp(lst) for x in lst]

def logsumexp(array):
    return gu.logsumexp(array)

def merge_append_dict(dict1, dict2):
    all_keys = unique_list(dict1.keys() + dict2.keys())
    out_dict = {}
    for i in all_keys:
        out_dict[i] = dict1.get(i, []) + dict2.get(i, [])
    return out_dict

def unique_list(lst):
    return list(set(lst))

def transpose_list(lst):
    return map(list, zip(*lst))

def max_empty(lst):
    if not lst:
        return 0
    else:
        return max(lst)

def list_to_dict(lst):
    """
    Transforms lists or arrays to dict.
    If list is nested (or array has more than one columns),
    the output is of the form dict{row: dict{col: value}}.
    Otherwise, output is dict{col:value}.
    """
    if hasattr(lst[0], '__iter__'):
        d = {j: {i: lst[j][i] for i in range(len(lst[j]))}
             for j in range(len(lst))}
    else:
        d = {i: lst[i] for i in range(len(lst))}
    return d

def list_intersection(l1, l2):
        return [x for x in l1 if x in l2]
 
