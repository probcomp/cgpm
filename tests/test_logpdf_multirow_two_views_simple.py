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

data_simple = [[1, 1]]

@pytest.fixture(scope="session")
def crosscat_simple():
    seed = 7
    data = data_simple
    D = len(data[0])
    outputs = range(D)
    Zv = OrderedDict([(0, 0), (1, 1)])
    Zr = [0]
    Zrv = {0: Zr, 1: Zr}
    view_alphas = [1., 1.]
    cctypes = ['categorical']*D
    distargs = {i: {'k': 2} for i in range(D)}
    hypers = {i: {'alpha': 1.} for i in range(D)}
    cc_2views = State(
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
    return cc_2views


def test_logpdf(crosscat_simple):
    # P(row_t = [1] | z_t = 0) = 7./12
    math_out = 2 * np.log(7. / 12)
    test_out = crosscat_simple.logpdf(-1, {0: 1, 1: 1})
    assert np.allclose(math_out, test_out)

def test_logpdf_given_one_row(crosscat_simple):
    # P(row_t=[1] | row_q1=[1])
    math_out = 2 * np.log(11./42 + 8./21)
    test_out = crosscat_simple.logpdf_multirow(-1, {0: 1, 1: 1}, {0: {0: 1, 1: 1}})
    assert np.allclose(math_out, test_out)

def test_logpdf_given_two_rows(crosscat_simple):
    # variables defined at
    # https://docs.google.com/document/d/1CvdNS6BuLiT7n8ERsNGcNjj-iNFODOCRz5JmI7kMKwI/edit

    # P(row_t=[1] | row_q1=[1], row_q2=[1])
    A = [4./7, 3./7]
    B1 = normalize_list([1./2, 1./6])
    B2 = normalize_list([2./9, 2./9, 1./6])
    B = B1+B2
    crp_posterior = A+B

    C1 = [12./20 + 1./8]
    C2 = [6./16 + 2./12 + 1./8]
    C5 = [1./2 + 1./8]
    conditional_predictive = C1 + C2*3 + C5
    C = conditional_predictive

    B_C = [B[i]*C[i] for i in xrange(5)]
    #     A[0]*(B[0]*C[0] + B[1]C[1])
    #     + A[1]*(B[2]*C[2] + B[3]*C[3] + B[4]*C[4])
    math_out = 2 * np.log(
        A[0]*(B_C[0]+B_C[1]) + A[1]*sum(B_C[2:]))
    test_out = crosscat_simple.logpdf_multirow(
        -1, {0: 1, 1: 1}, {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}})
    
    assert np.allclose(math_out, test_out)
    
def normalize_list(lst):
    sum_lst = sum(lst)
    return [float(i)/sum_lst for i in lst]
