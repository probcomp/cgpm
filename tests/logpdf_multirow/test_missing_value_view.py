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
from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp

@pytest.fixture(params=[(seed, D, do_analyze) 
                        for seed in range(1) 
                        for D in [2]
                        for do_analyze in [False]])
def exampleCGPM(request):
    seed, D, do_analyze = request.param

    rng = gu.gen_rng(seed)
    data = rng.choice([0, 1], size=(100, D))
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(
        X,
        cctypes=['bernoulli']*D,
        outputs=[1000] + outputs,
        rng=rng)

    view.seed = seed
    view.D = D
    if do_analyze:
        view.analyze(N=30)
    return view
 
def test_crash_incorporate_missing_value(exampleCGPM):
    view = exampleCGPM

    if view.D == 2:
        # 1. incorporate full row and sample cluster
        normal_row = {i: 1 for i in range(2)}
        view.incorporate(42000, query=normal_row)
        view.unincorporate(42000)

        # 2. incorporate full row and force cluster
        forced_row = {i: 1 for i in range(2) + [view.outputs[0]]}
        view.incorporate(42001, query=forced_row)
        view.unincorporate(42001)

        # 3. incorporate row with missing value 
        missing_row = {0: 1}
        view.incorporate(42002, query=missing_row)
        view.unincorporate(42002)

# view.logpdf already accepts query with missing values
# def test_logpdf_missing_value(exampleCGPM):
#     view = exampleCGPM
 
#     if view.D == 2:
#         # 1. Test crash for missing value logpdf
#         marg_query = {0: 1}
#         marg_logpdf = view.logpdf(-1, query=marg_query)  # should crash
#         assert False, "marg_logpdf should have crashed"

#         # 2. Test missing value logpdf makes sense
#         joint_queries = [{0: 1, 1: i} for i in range(2)]
#         joint_logpdfs = [view.logpdf(-1, query=joint_queries[i])
#                          for i in range(2)]
#         marg_joint_logpdf = logsumexp(joint_logpdfs)

#         assert np.allclose(marg_logpdf, marg_joint_logpdf, atol=.1)

#     else:
#         raise Exception("No test run")

def test_logpdf_multirow_missing_value(exampleCGPM):
    view = exampleCGPM
    
    if view.D == 2:
        # 0. Missing value in query should run
        marg_query = {0: 1}
        evid = {0: {0: 1, 1: 1}}
        lp0 = view.logpdf_multirow(
            -1, query=marg_query, evidence=evid)

        # 1. Test crash for missing value in evidence
        query = {0: 1, 1: 1}
        marg_evid = {0: {0: 1}}
        lp1 = view.logpdf_multirow(
            -1, query=query, evidence=marg_evid)

        # 2. Test missing value logpdf marginalizes correctly on query
        # # sum_i P(q=[1,i]| e=[1,nan]) = P(q=[1,nan]| e=[1,nan])
        joint_queries = [{0: 1, 1: i} for i in range(2)]
        joint_logpdfs = [view.logpdf_multirow(
            -1, query=joint_queries[i], evidence=marg_evid) for i in range(2)]
        lp_left2 = logsumexp(joint_logpdfs)
        lp_right2 = view.logpdf_multirow(
            -1, query={0: 1}, evidence={0: {0: 1}})
        assert np.allclose(lp_left2, lp_right2, atol=.1)

        # 3. Test missing value logpdf marginalizes correctly on evidence
        # # sum_i P(q=[1, nan]| e=[i, nan]) P(e=[i, nan]) = P(q=[1,nan])
        marg_lps = [view.logpdf(-1, query={0: i}) for i in range(2)]
        lp_right3 = marg_lps[1]
        cond_lps = [
            view.logpdf_multirow(-1, query=marg_query, evidence={0: {0: i}}) +
            view.logpdf(-1, query={0: i}) for i in range(2)]
        lp_left3 = logsumexp(cond_lps)
        assert np.allclose(lp_left3, lp_right3, atol=.1)

    else:
        raise Exception("No test run")

# def test_simulate_missing_value():
#     raise NotImplementedError
