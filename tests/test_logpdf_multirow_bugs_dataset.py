import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn
import os 

from collections import OrderedDict

from cgpm.crosscat.state import State
from cgpm.crosscat.engine import Engine
from cgpm.mixtures.view import View
from cgpm.utils import bayessets_utils as bu
from cgpm.bayessets import bayes_sets as bs
from cgpm.utils import render_utils as ru

seaborn.set_style("white")
PKLDIR = 'tests/resources/pkl/'
DATADIR = 'tests/resources/data/'
OUT = 'tests/resources/out/'

feature_names = "Tail Pattern Head Antenna Feet Legs".split()
row_names = [str(i) for i in range(1, 9)]
 
data_3_3 = [[0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0]]

data_3_2 = [[0, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0]]

data_2 = [[1, 1, 0, 1, 1, 1],
          [1, 1, 1, 0, 1, 1],
          [1, 1, 1, 1, 0, 1],
          [1, 1, 1, 1, 1, 0],
          [1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0]]

@pytest.fixture(scope="session")
def dpmm_2clusters():
    seed = 7
    data = data_2
    D = len(data[0])
    outputs = range(D)
    Zv = OrderedDict((i, 0) for i in outputs)
    Zr = [0 if i < 4 else 1 for i in range(8)]
    Zrv = {0: Zr}
    view_alphas = [1.]
    cctypes = ['categorical']*D
    distargs = {i: {'k': 2} for i in range(D)}
    hypers = {i: {'alpha': 1.} for i in range(D)}
    dpmm_2clusters = State(
        data,
        outputs=outputs,
        cctypes=cctypes,
        rng=np.random.RandomState(seed),
        distargs=distargs,
        hypers=hypers,
        Zv=Zv,
        Zrv=Zrv,
        view_alphas=view_alphas
    )
    return dpmm_2clusters

def test_logpdf_given_cluster_one_col(dpmm_2clusters):
    dim = dpmm_2clusters.views[0].dims[0]

    # P(col_0=1 | z=0) = 0.833
    math_out = np.log(5. / 6.)
    test_out = dim.logpdf(-1, {0: 1}, {10000000: 0})
    assert np.allclose(math_out, test_out)

    # P(col_0=1 | z=1) = 0.333
    math_out = np.log(2. / 6.)
    test_out = dim.logpdf(-1, {0: 1}, {10000000: 1})
    assert np.allclose(math_out, test_out)

def test_logpdf_given_cluster_all_cols(dpmm_2clusters):
    # P(x=[1,1,1,1,1,1] | z=0) 
    math_out = sum(np.log(
        [5./6, 5./6, 4./6, 4./6, 4./6, 4./6]))
    test_out = dpmm_2clusters.logpdf(-1, {i: 1 for i in range(6)},
                                     {10000000: 0})
    assert np.allclose(math_out, test_out)

    # P(x=[1,1,1,1,1,1] | z=1)
    math_out = sum(np.log(
        [2./6, 2./6, 2./6, 2./6, 1./6, 1./6]))
    test_out = dpmm_2clusters.logpdf(-1, {i: 1 for i in range(6)},
                                     {10000000: 1})
    assert np.allclose(math_out, test_out)

def test_logpdf_one_col(dpmm_2clusters):
    # P(col_0=1)
    math_out = np.log(31./54)
    test_out = dpmm_2clusters.logpdf(-1, {0: 1})
    assert np.allclose(math_out, test_out)
    
def test_logpdf_all_cols(dpmm_2clusters):
    # P(x=[1,1,1,1,1,1])
    A = np.prod([5./6, 5./6, 4./6, 4./6, 4./6, 4./6])
    B = np.prod([2./6, 2./6, 2./6, 2./6, 1./6, 1./6])
    C = .5 ** 6
    math_out = np.log(4./9 * (A + B) + C/9)
    test_out = dpmm_2clusters.logpdf(-1, {i: 1 for i in range(6)})
    assert np.allclose(math_out, test_out)

def test_logpdf_posterior_crp(dpmm_2clusters):
    # First math attempt
    weight_zy_given_y = [20./54, 8./54, 3./54]
    prob_zy_given_y = np.array(weight_zy_given_y) / sum(weight_zy_given_y)
    math_out_1 = prob_zy_given_y

    # Second math attempt
    view = dpmm_2clusters.views[0]
    # likelihood params
    dirichlet_beta = view.dims[0].hypers['alpha']
    clusters = view.dims[0].clusters
    counts_one = [clusters[0].counts[1],
                  clusters[1].counts[1],
                  0]
    N_cluster = [sum(clusters[0].counts),
                 sum(clusters[1].counts),
                 0]
    likelihood = [
        (dirichlet_beta + counts_one[i]) / (2*dirichlet_beta + N_cluster[i])
        for i in range (3)]
    
    # prior params
    crp_alpha = view.crp.hypers['alpha']
    Nk = view.Nk()
    Ntotal = sum(Nk.values())
    crp_numerator = [Nk[i] for i in range(2)] + [crp_alpha]
    prior = [crp_numerator[i] / (crp_alpha + Ntotal) for i in range(3)]

    weight_zy_given_y_second = [likelihood[i] * prior[i] for i in range(3)]
    assert np.allclose(weight_zy_given_y, weight_zy_given_y_second)

def test_posterior_crp_logpdf_multirow_one_col(dpmm_2clusters):
    weight_zy_given_y = [20./54, 8./54, 3./54]
    prob_zy_given_y = np.array(weight_zy_given_y) / sum(weight_zy_given_y)
    math_out = prob_zy_given_y

    # set a trace on view.logpdf_multirow (line 299) 
    view = dpmm_2clusters.views[0]
    view.logpdf_multirow(-1, {0: 0}, {0: {0: 1}}, debug=True)
    test_out = np.exp(view.debug['posterior_crp_logpdf'])
    assert np.allclose(math_out, test_out)
    del view.debug
    # compare weight_zy_given_y with lp_evidence_unorm
    # same values, no bug there

def test_posterior_query_logpdf_multirow_one_col(dpmm_2clusters):
    prob_x_given_zx_zy_y = np.array([
        [6./7, 2./6, 1./2, 0],
        [5./6, 3./7, 1./2, 0],
        [5./6, 2./6, 2./3, 1./2]])
    prob_zx_given_zy = np.array([
        [5./10, 4./10, 1./10, 0],
        [4./10, 5./10, 1./10, 0],
        [4./10, 4./10, 1./10, 1./10]])

    prob_x_zx_given_zy_y = prob_x_given_zx_zy_y * prob_zx_given_zy
    math_out = prob_x_zx_given_zy_y.sum(axis=1)

    view = dpmm_2clusters.views[0]
    view.logpdf_multirow(-1, {0: 0}, {0: {0: 1}}, debug=True)
    test_out = np.exp(view.debug['conditional_logpredictive'])

    assert np.allclose(math_out, test_out)
    del view.debug

    # BUG

def test_logpdf_multirow_one_col(dpmm_2clusters):
    # P(x = 1 | y = 1) 
    #  = sum_{i,j}  P(zy=i | y=1) P(zx=j | zy=i)  P(x=1| zx=j, zy=i, y=1)
    # # P(zy = i | y = 1)
    weight_zy_given_y = [20./54, 8./54, 3./54]
    prob_zy_given_y = np.array(weight_zy_given_y) / sum(weight_zy_given_y)
    
    # # P(zx = j | zy = i)
    prob_zx_given_zy = np.array([
        [5./10, 4./10, 1./10, 0],
        [4./10, 5./10, 1./10, 0],
        [4./10, 4./10, 1./10, 1./10]])

    # # P(x = 1| zx = j, zy = i, y = 1)
    prob_x_given_zx_zy_y = np.array([
        [6./7, 2./6, 1./2, 0],
        [5./6, 3./7, 1./2, 0],
        [5./6, 2./6, 2./3, 1./2]])

    math_out = np.log(sum(
        [prob_x_given_zx_zy_y[i, j] * prob_zx_given_zy[i, j] * prob_zy_given_y[i]
         for i in range(3) for j in range(4)]))
    test_out = dpmm_2clusters.logpdf_multirow(
        -1, {0: 1}, {0: {0: 1}})  
    assert np.allclose(math_out, test_out)

    # # P(x = 0| zx = j, zy = i, y = 1)
    prob_x_given_zx_zy_y = np.array([
        [1./7, 4./6, 1./2, 0],
        [1./6, 4./7, 1./2, 0],
        [1./6, 4./6, 1./3, 1./2]])
    math_out = np.log(sum(
        [prob_x_given_zx_zy_y[i, j] * prob_zx_given_zy[i, j] * prob_zy_given_y[i]
         for i in range(3) for j in range(4)]))
    test_out = dpmm_2clusters.logpdf_multirow(
        -1, {0: 0}, {0: {0: 1}})  
    assert np.allclose(math_out, test_out)
