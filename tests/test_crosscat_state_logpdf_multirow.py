# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This test suite ensures that results returned from View.logpdf_multirow,
State.logpdf_multirow and Engine.logpdf_multirow, are analytically correct
(for some cases) and follow the rules of probability. 
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu
from cgpm.utils.general import logsumexp

from cgpm.crosscat.state import State

OUT = '../images/'

@pytest.fixture
def priorCGPM():
    data = np.random.choice([0, 1], size=(100, 5))
    outputs = range(5)
    rng=gu.gen_rng(0)
    state = State(data, cctypes=['bernoulli']*5, rng=rng)
    return state

@pytest.fixture
def posteriorCGPM():
    state = priorCGPM()
    state.transition(N=10)
    return state

def test_logpdf_multirow_analytical(priorCGPM):
    dpmbb = priorCGPM
    
    # case 1: p(x_1=1|x_0=1)
    out_1 = dpmbb.logpdf_multirow(-1, query={0: 1}, evidence={0: {0: 1}})
    analytical_1 = np.log(0.569)  # computed by hand
    assert np.allclose(analytical_1, out_1, atol=0.1)

    # case 2: p(x_1=[1,1,1,1,1]| x_0=[1,1,1,1,1])
    x1 = {i: 1 for i in range(5)}
    x0 = {0: x1}
    out_2 = dpmbb.logpdf_multirow(-1, query=x1, evidence=x0)
    analytical_2 = np.log(0.134)  # computed by hand
    assert np.allclose(analytical_2, out_2, atol=0.1)

def test_logpdf_multirow_marginalization(priorCGPM, posteriorCGPM):
    """ 
    Check p(x2) = sum_x1 (p(x2|x1) p(x1)), or equivalently
      log(p(x2) = logsumexp_x1 (logp(x2|x1) + logp(x1))
    """

    log_marginal_i = lambda d: priorCGPM.logpdf(-1, query=d)
    log_conditional_i = lambda d1, d2: priorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    log_marginal_t = lambda d: posteriorCGPM.logpdf(-1, query=d)
    log_conditional_t = lambda d1, d2: posteriorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    # case 1: empty cgpm, x_2 = 1
    dx2 = {0: 1}
    dx1 = [{0: {0: 0}}, {0: {0: 1}}]
    left_1 = log_marginal_i(dx2) 
    summand_1 = [
        log_marginal_i(d.values()[0]) + log_conditional_i(dx2, d) for d in dx1]
    assert np.allclose(left_1, logsumexp(summand_1))

    # case 2: empty cgpm, x_2 = 0
    dx2 = {0: 0}
    left_2 = log_marginal_i(dx2) 
    summand_2 = [
        log_marginal_i(d.values()[0]) + log_conditional_i(dx2, d) for d in dx1]
    assert np.allclose(left_1, logsumexp(summand_1))
    
    # check whether p(x2) is probability
    assert np.allclose(logsumexp([left_1, left_2]), 0)
 
    # case 3: cgpm with data, x_2 = 1
    dx2 = {0: 1}
    left_3 = log_marginal_t(dx2) 
    summand_3 = [
        log_marginal_t(d.values()[0]) + log_conditional_t(dx2, d) for d in dx1]
    assert np.allclose(left_1, logsumexp(summand_1))

    # case 4: cgpm with, x_2 = 0
    dx2 = {0: 0}
    left_4 = log_marginal_t(dx2) 
    summand_4 = [
        log_marginal_t(d.values()[0]) + log_conditional_t(dx2, d) for d in dx1]
    assert np.allclose(left_1, logsumexp(summand_1))
    
    # check whether p(x2) is probability
    assert np.allclose(logsumexp([left_3, left_4]), 0)

    # case 5: cgpm with data, x_2 = [1,1,1,1,1]
    dx2 = {i: 1 for i in range(5)}
    dx1 = [{0: {0: i, 1: j, 2: k, 3: l, 4: m}} for i, j, k, l, m in product(
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])]
    left_5 = log_marginal_t(dx2) 
    summand_5 = [
        log_marginal_t(d.values()[0]) * log_conditional_t(dx2, d) for d in dx1]
    assert np.allclose(left_1, logsumexp(summand_1))

def test_logpdf_multirow_bayes(priorCGPM, posteriorCGPM):
    """
    Check whether p(x2|x1) p(x2) = p(x1|x2)p(x1)
    """
    log_marginal_i = lambda d: priorCGPM.logpdf(-1, query=d)
    log_conditional_i = lambda d1, d2: priorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    log_marginal_t = lambda d: posteriorCGPM.logpdf(-1, query=d)
    log_conditional_t = lambda d1, d2: posteriorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    # case 1: x2=1, x1=0
    dx1 = {0: {0: 0}}
    dx2 = {0: {0: 1}}
    left_1 = log_conditional_i(dx2[0], dx1) 
    right_1 = (log_conditional_i(dx1[0], dx2) + log_marginal_i(dx2[0]) - 
               log_marginal_i(dx1[0]))
    assert np.allclose(left_1, right_1)

    # case 2: x2=0, x1=0
    dx2 = {0: {0: 0}}
    left_2 = log_conditional_i(dx2[0], dx1) 
    right_2 = (log_conditional_i(dx1[0], dx2) + log_marginal_i(dx2[0]) - 
               log_marginal_i(dx1[0]))
    assert np.allclose(left_2, right_2)

    # check whether p(x2|x1=0) is probability
    assert np.allclose(logsumexp([left_1, left_2]), 0)

    # case 3: incorporate data; x2=1, x1=0
    dx2 = {0: {0: 1}}
    left_3 = log_conditional_t(dx2[0], dx1) 
    right_3 = (log_conditional_t(dx1[0], dx2) + log_marginal_t(dx2[0]) - 
               log_marginal_t(dx1[0]))
    assert np.allclose(left_3, right_3)

    # case 4: x2=0, x1=0
    dx2 = {0: {0: 0}}
    left_4 = log_conditional_t(dx2[0], dx1) 
    right_4 = (log_conditional_t(dx1[0], dx2) + log_marginal_t(dx2[0]) - 
               log_marginal_t(dx1[0]))
    assert np.allclose(left_4, right_4)

    # check whether p(x2|x1=0) is probability
    assert np.allclose(logsumexp([left_3, left_4]), 0)
    
    # case 5: cgpm with data, x_2 = [1,1,1,1,1], 
    dx2 = {0: {i: 1 for i in range(5)}}
    dx1 = {0: {i: 0 for i in range(5)}}
    left_5 = log_conditional_t(dx2[0], dx1) 
    right_5 = (log_conditional_t(dx1[0], dx2) + log_marginal_t(dx2[0]) - 
               log_marginal_t(dx1[0]))
    assert np.allclose(left_5, right_5)

def test_logpdf_multirow_plot(priorCGPM):
    dpmbb = priorCGPM
    
    query = {0: 1}
    d = {} 
    logp = [dpmbb.logpdf(-1, query=query)]
    for i in range(10):
        d[i] = {0: 1}
        logp.append(dpmbb.logpdf_multirow(-1, query=query, evidence=d))
    Nx = range(len(logp))
    plt.plot(Nx, logp)
    plt.xlabel("$n$ (number of conditioned rows)")
    plt.ylabel("$\log(p(x_t=1|x_1=1,\ldots,x_n=1))$")
    plt.savefig(OUT+"test_logpdf_multirow_univariate_increasing_evidence.png")
