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

OUT = 'tests/resources/out'

@pytest.fixture(scope="session", params=[
    (seed, D, do_analyze) for seed in range(3) 
    for D in [1, 5] for do_analyze in [False, True]])
def exampleCGPM(request):
    seed = request.param[0]
    D = request.param[1]
    rng = gu.gen_rng(seed)
    data = rng.choice([0, 1], size=(100, D))
    state = State(data, cctypes=['bernoulli']*D, rng=rng)
    state.seed = seed
    if do_analyze:
        analyze_until_view_partitions(state)
    return state
    
def test_marginalization(exampleCGPM):
    """ 
    Check p(x2) = sum_x1 (p(x2|x1) p(x1)), or equivalently
      log(p(x2) = logsumexp_x1 (logp(x2|x1) + logp(x1))
    """

    D = len(exampleCGPM.outputs)

    log_marginal = lambda d: exampleCGPM.logpdf(-1, query=d)
    log_conditional = lambda d1, d2: exampleCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    if D == 1:
        # case 1: empty cgpm, x_2 = 1
        dx2 = {0: 1}
        dx1 = [{0: {0: 0}}, {0: {0: 1}}]
        left_1 = log_marginal(dx2) 
        summand_1 = [
            log_marginal(d.values()[0]) + log_conditional(dx2, d)
            for d in dx1]
        assert np.allclose(left_1, logsumexp(summand_1), atol=0.1)

        # case 2: empty cgpm, x_2 = 0
        dx2 = {0: 0}
        left_2 = log_marginal(dx2) 
        summand_2 = [
            log_marginal(d.values()[0]) + log_conditional(dx2, d)
            for d in dx1]
        assert np.allclose(left_2, logsumexp(summand_2), atol=0.1)

        # check whether p(x2) is probability
        assert np.allclose(logsumexp([left_1, left_2]), 0, atol=0.1)

    # case 5: posterior cgpm, x_2 = [1] * D
    elif D == 5:
        dx2 = {i: 1 for i in range(D)}
        dx1 = [{0: {0: i, 1: j, 2: k, 3: l, 4: m}}
               for i, j, k, l, m in product(
               [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])]
        left_3 = log_marginal(dx2) 
        summand_3 = [
            log_marginal(d.values()[0]) + log_conditional(dx2, d)
            for d in dx1]
        assert np.allclose(left_3, logsumexp(summand_3), atol=0.1)
 
    else:
        assert False, "No test was called"

def test_bayes(exampleCGPM):
    """
    Check whether p(x2|x1)= p(x1|x2)p(x2)/ p(x1) 
          or log(p(x2|x1)) = log(p(x1|x2)) + log(p(x2)) - log(p(x1))
    """

    log_marginal = lambda d: exampleCGPM.logpdf(-1, query=d)
    log_conditional = lambda d1, d2: exampleCGPM.logpdf_multirow(-1, 
        query=d1, evidence=d2)

    D = len(exampleCGPM.outputs)

    if D == 1:
        # case 1: x2=1, x1=0
        dx1 = {0: {0: 0}}
        dx2 = {0: {0: 1}}
        left_1 = log_conditional(dx2[0], dx1) 
        right_1 = (log_conditional(dx1[0], dx2) + log_marginal(dx2[0]) - 
                   log_marginal(dx1[0]))
        assert np.allclose(left_1, right_1, atol=0.1)

        # case 2: x2=0, x1=0
        dx2 = {0: {0: 0}}
        left_2 = log_conditional(dx2[0], dx1) 
        right_2 = (log_conditional(dx1[0], dx2) + log_marginal(dx2[0]) - 
                   log_marginal(dx1[0]))
        assert np.allclose(left_2, right_2, atol=0.1)

        # check whether p(x2|x1=0) is probability
        assert np.allclose(logsumexp([left_1, left_2]), 0, atol=0.1)

    else:
        dx2 = {0: {i: 1 for i in range(D)}}
        dx1 = {0: {i: 0 for i in range(D)}}
        left_3 = log_conditional(dx2[0], dx1) 
        right_3 = (log_conditional(dx1[0], dx2) + log_marginal(dx2[0]) - 
                   log_marginal(dx1[0]))
        assert np.allclose(left_3, right_3)

# def test_independent_views(exampleCGPM):

#     state = exampleCGPM
#     assert len(state.views) > 1, "test only suited for multiple views"

#     poststate = analyze_until_view_partitions(state)
#     full_query, full_evid = get_full_query_evidence(poststate)
#     part_query, part_evid = get_view_partitioned_query_evidence(poststate)

#     full_logp = poststate.logpdf_multirow(-1, full_query, full_evid)
#     part_logp = [poststate.logpdf_multirow(-1, part_query[i], part_evid[i])
#                  for i in range(len(part_query))]

#     assert np.allclose(full_logp, sum(part_logp))

#     part_full_logp = [poststate.logpdf_multirow(-1, part_query[i], full_evid)
#                       for i in range(len(part_query))]
#     assert part_logp == part_full_logp



def test_generative_logscore(exampleCGPM):
    state = exampleCGPM
    D = len(state.outputs) 

    query = {i: 0 for i in range(D)}
    evidence = {0: query}
    
    assert state.generative_logscore(query, evidence) < 40

 ## HELPERS ##

def analyze_until_view_partitions(model):
    # While there is only one view, analyze_until_view_partitions model
    i = 0
    while len(set(model.Zv().values())) == 1 and i < 10:
        model.transition(N=10)
        i += 1
    return model

def get_view_partitioned_query_evidence(model):
    q = []
    ev = []
    for key, view in model.views.iteritems():
        view_outputs = view.outputs[1:]
        q.append({i: 1 for i in view_outputs})
        # Is this column in the view? If so, add it, otherwise, ignore it.
        ev.append({j: {i: j for i in view_outputs} for j in range(2)})
    return q, ev

def get_full_query_evidence(model):
    D = len(model.dims())
    q = {i: 1 for i in range(D)}
    ev = {j: {i: j for i in range(D)} for j in range(2)}
    return q, ev
