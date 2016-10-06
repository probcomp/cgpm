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
# RUN TESTS IN cgpm/ DIRECTORY
 
import pytest
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu
from cgpm.utils.general import logsumexp

from cgpm.mixtures.view import View

OUT = 'tests/resources/out/'

@pytest.fixture(params=[(seed, D) for seed in range(3) for D in [1, 5]])
def priorCGPM(request):
    seed = request.param[0]
    D = request.param[1]
    rng = gu.gen_rng(seed)
    data = rng.choice([0, 1], size=(100, D))
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    model = View(
        X,
        cctypes=['bernoulli']*D,
        outputs=[1000] + outputs,
        rng=rng)
    model.seed = seed
    return model

def analyze(model):
    model.transition(N=10)
    return model

# Broken. To fix, do:
#  - Create View with empty dataset
#  - Force Beta Bernoulli hyperparameters to [.1, .1] 
#    - Alternatively, rederive using hyperprior.

# def test_logpdf_multirow_analytical(priorCGPM):
#     view = priorCGPM
    
#     # case 1: p(x_1=1|x_0=1)
#     out_1 = view.logpdf_multirow(-1, query={0: 1}, evidence={0: {0: 1}})
#     analytical_1 = np.log(0.569)  # computed by hand
#     assert np.allclose(analytical_1, out_1, atol=0.1)

#     # case 2: p(x_1=[1,1,1,1,1]| x_0=[1,1,1,1,1])
#     x1 = {i: 1 for i in range(5)}
#     x0 = {0: x1}
#     out_2 = view.logpdf_multirow(-1, query=x1, evidence=x0)
#     analytical_2 = np.log(0.134)  # computed by hand
#     assert np.allclose(analytical_2, out_2, atol=0.1)


def test_logpdf_multirow_marginalization(priorCGPM):
    """ 
    Check p(x2) = sum_x1 (p(x2|x1) p(x1)), or equivalently
      log(p(x2) = logsumexp_x1 (logp(x2|x1) + logp(x1))
    """
    posteriorCGPM = analyze(priorCGPM)
    D = len(priorCGPM.outputs) - 1

    log_marginal_i = lambda d: priorCGPM.logpdf(-1, query=d)
    log_conditional_i = lambda d1, d2: priorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    log_marginal_t = lambda d: posteriorCGPM.logpdf(-1, query=d)
    log_conditional_t = lambda d1, d2: posteriorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    if D == 1:
        # case 1: empty cgpm, x_2 = 1
        dx2 = {0: 1}
        dx1 = [{0: {0: 0}}, {0: {0: 1}}]
        left_1 = log_marginal_i(dx2) 
        summand_1 = [
            log_marginal_i(d.values()[0]) + log_conditional_i(dx2, d)
            for d in dx1]
        assert np.allclose(left_1, logsumexp(summand_1))

        # case 2: empty cgpm, x_2 = 0
        dx2 = {0: 0}
        left_2 = log_marginal_i(dx2) 
        summand_2 = [
            log_marginal_i(d.values()[0]) + log_conditional_i(dx2, d)
            for d in dx1]
        assert np.allclose(left_2, logsumexp(summand_2))

        # check whether p(x2) is probability
        assert np.allclose(logsumexp([left_1, left_2]), 0)

        # case 3: cgpm with data, x_2 = 1
        dx2 = {0: 1}
        left_3 = log_marginal_t(dx2) 
        summand_3 = [
            log_marginal_t(d.values()[0]) + log_conditional_t(dx2, d)
            for d in dx1]
        assert np.allclose(left_3, logsumexp(summand_3))

        # case 4: cgpm with, x_2 = 0
        dx2 = {0: 0}
        left_4 = log_marginal_t(dx2) 
        summand_4 = [
            log_marginal_t(d.values()[0]) + log_conditional_t(dx2, d)
            for d in dx1]
        assert np.allclose(left_4, logsumexp(summand_4))

        # check whether p(x2) is probability
        assert np.allclose(logsumexp([left_3, left_4]), 0)

    # case 5: posterior cgpm, x_2 = [1] * D
    if D == 5:
        dx2 = {i: 1 for i in range(D)}
        dx1 = [{0: {0: i, 1: j, 2: k, 3: l, 4: m}}
               for i, j, k, l, m in product(
               [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])]
        left_5 = log_marginal_t(dx2) 
        summand_5 = [
            log_marginal_t(d.values()[0]) + log_conditional_t(dx2, d)
            for d in dx1]
        assert np.allclose(left_5, logsumexp(summand_5))


def test_logpdf_multirow_bayes(priorCGPM):
    """
    Check whether p(x2|x1)= p(x1|x2)p(x2)/ p(x1) 
          or log(p(x2|x1)) = log(p(x1|x2)) + log(p(x2)) - log(p(x1))
    """
    posteriorCGPM = analyze(priorCGPM)

    log_marginal_i = lambda d: priorCGPM.logpdf(-1, query=d)
    log_conditional_i = lambda d1, d2: priorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    log_marginal_t = lambda d: posteriorCGPM.logpdf(-1, query=d)
    log_conditional_t = lambda d1, d2: posteriorCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    D = len(priorCGPM.outputs)-1

    if D == 1:  # CGPM with two columns
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

    # case 5: posterior cgpm, x_2 = [1]*D, 
    dx2 = {0: {i: 1 for i in range(D)}}
    dx1 = {0: {i: 0 for i in range(D)}}
    left_5 = log_conditional_t(dx2[0], dx1) 
    right_5 = (log_conditional_t(dx1[0], dx2) + log_marginal_t(dx2[0]) - 
               log_marginal_t(dx1[0]))
    assert np.allclose(left_5, right_5)


# Broken. To fix, do:
#  - Let view incorporate rows with missing values.
#  - [X] Alternatively, hack new plot.
# def test_logpdf_multirow_plot(priorCGPM):
#     view = priorCGPM

#     D = len(view.outputs)-1
#     query = {0: 1 for i in range(D)}
#     d = {} 
#     logp = [view.logpdf(-1, query=query)]
#     for i in range(3):
#         d[i] = {i: 1 for i in range(D)}
#         logp.append(view.logpdf_multirow(-1, query=query, evidence=d))
#     Nx = range(len(logp))
#     plt.plot(Nx, logp)
#     plt.xlabel("$n$ (number of conditioned rows)")
#     plt.ylabel("$\log(p(x_t=1|x_1=1,\ldots,x_n=1))$")
#     plt.savefig(OUT + '''
#     test_logpdf_multirow_univariate_increasing_seed%d_%dD_evidence.png'''
#                 % (priorCGPM.seed, D))
#     plt.clf()

def test_generative_logscore(priorCGPM):
    posteriorCGPM = analyze(priorCGPM)
    view = posteriorCGPM
    D = len(view.outputs) 

    query = {i: 0 for i in range(D)}
    evidence = {0: query}
    
    view.generative_logscore(query, evidence)
