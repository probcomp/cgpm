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

import pytest
import numpy as np

from cgpm.utils import general as gu
from cgpm.crosscat.state import State
from cgpm.utils.general import logsumexp

@pytest.fixture(scope="session", params=[(1,2,0)])
    # (seed, D, do_analyze) for seed in range(3) 
    # for D in [1, 5] for do_analyze in [False, True]])
def exampleCGPM(request):
    seed, D, do_analyze = request.param

    rng = gu.gen_rng(seed)
    data = rng.choice([0, 1], size=(100, D))
    state = State(data, cctypes=['bernoulli']*D, rng=rng)

    state.seed = seed
    state.D = D
    if do_analyze:
        analyze_until_view_partitions(state)
    return state
    
def test_crash_incorporate_missing_value(exampleCGPM):
    # state = exampleCGPM

    # # incorporate full row and sample cluster (works)
    # normal_row = {i: 1 for i in range(5)}
    # state.incorporate(42000, query=normal_row)
    # state.unincorporate(42000)

    # # incorporate full row and force cluster (works)
    # forced_row = {i: 1 for i in range(5) + [state.outputs[0]]}
    # state.incorporate(42001, query=forced_row)
    # state.unincorporate(42001)

    # # incorporate row with missing values (fails)
    # missing_row = {i: 1 for i in range(3)}
    # state.incorporate(42002, query=missing_row)
    # state.unincorporate(42002)
    raise NotImplementedError

def test_crash_incorporate_missing_value(exampleCGPM):
    state = exampleCGPM

    if state.D == 2:
        # 1. incorporate full row and sample cluster (works)
        normal_row = {i: 1 for i in range(2)}
        state.incorporate(42000, query=normal_row)
        state.unincorporate(42000)

        # 2. incorporate full row and force cluster (works)
        forced_row = {i: 1 for i in range(2) + [state.outputs[0]]}
        state.incorporate(42001, query=forced_row)
        state.unincorporate(42001)

        # 3. incorporate row with missing value (fails)
        missing_row = {0: 1}
        state.incorporate(42002, query=missing_row)
        state.unincorporate(42002)
        assert False, "should have crashed"

# state.logpdf already accepts query with missing values
# def test_logpdf_missing_value(exampleCGPM):
#     state = exampleCGPM
 
#     if state.D == 2:
#         # 1. Test crash for missing value logpdf
#         marg_query = {0: 1}
#         marg_logpdf = state.logpdf(-1, query=marg_query)  # should crash
#         assert False, "marg_logpdf should have crashed"

#         # 2. Test missing value logpdf makes sense
#         joint_queries = [{0: 1, 1: i} for i in range(2)]
#         joint_logpdfs = [state.logpdf(-1, query=joint_queries[i])
#                          for i in range(2)]
#         marg_joint_logpdf = logsumexp(joint_logpdfs)

#         assert np.allclose(marg_logpdf, marg_joint_logpdf, atol=.1)

#     else:
#         raise Exception("No test run")

def test_logpdf_multirow_missing_value(exampleCGPM):
    state = exampleCGPM
    
    if state.D == 2:
        # 0. Missing value in query should run
        marg_query = {0: 1}
        evid = {0: {0: 1, 1: 1}}
        lp0 = state.logpdf_multirow(
            -1, query=marg_query, evidence=evid)

        # 1. Test crash for missing value in evidence
        query = {0: 1, 1: 1}
        marg_evid = {0: {0: 1}}
        lp1 = state.logpdf_multirow(
            -1, query=query, evidence=marg_evid)
        assert False, "should have crashed"

        # 2. Test missing value logpdf marginalizes correctly on query
        # # sum_i P(q=[1,i]| e=[1,nan]) = P(q=[1,nan]| e=[1,nan])
        joint_queries = [{0: 1, 1: i} for i in range(2)]
        joint_logpdfs = [state.logpdf_multirow(
            -1, query=joint_queries[i], evidence=marg_evid) for i in range(2)]
        lp2 = logsumexp(joint_logpdfs)
        assert np.allclose(lp1, lp2, atol=.1)

        # 3. Test missing value logpdf marginalizes correctly on evidence
        # # sum_i P(q=[1, nan]| e=[i, nan]) P(e=[i, nan]) = P(q=[1,nan])
        marg_lps = [state.logpdf(-1, query={0: i}) for i in range(2)]
        lp_right = marg_lps[1]
        cond_lps = [
            state.logpdf_multirow(-1, query=marg_query, evidence={0: {0: i}}) +
            state.logpdf(-1, query={0: i}) for i in range(2)]
        lp_left = logsumexp(cond_lps)
        assert np.allclose(lp_left, lp_right, atol=.1)

    else:
        raise Exception("No test run")

def test_simulate_missing_value():
    raise NotImplementedError


## HELPERS ##

def analyze_until_view_partitions(model):
    # While there is only one state, analyze_until_view_partitions model
    i = 0
    while len(set(model.Zv().values())) == 1 and i < 10:
        model.transition(N=10)
        i += 1
    return model
