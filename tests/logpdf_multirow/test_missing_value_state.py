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

@pytest.fixture(scope="session", params=[(1,1,0)])
    # (seed, D, do_analyze) for seed in range(3) 
    # for D in [1, 5] for do_analyze in [False, True]])
def exampleCGPM(request):
    seed = request.param[0]
    D = request.param[1]
    do_analyze = request.param[2]

    rng = gu.gen_rng(seed)
    data = rng.choice([0, 1], size=(100, D))
    state = State(data, cctypes=['bernoulli']*D, rng=rng)
    state.seed = seed
    if do_analyze:
        analyze_until_view_partitions(state)
    return state
    
def test_crash_incorporate_missing_value(exampleCGPM):
    # view = exampleCGPM

    # # incorporate full row and sample cluster (works)
    # normal_row = {i: 1 for i in range(5)}
    # view.incorporate(42000, query=normal_row)
    # view.unincorporate(42000)

    # # incorporate full row and force cluster (works)
    # forced_row = {i: 1 for i in range(5) + [view.outputs[0]]}
    # view.incorporate(42001, query=forced_row)
    # view.unincorporate(42001)

    # # incorporate row with missing values (fails)
    # missing_row = {i: 1 for i in range(3)}
    # view.incorporate(42002, query=missing_row)
    # view.unincorporate(42002)
    raise NotImplementedError

def test_logpdf_missing_value(exampleCGPM):
    raise NotImplementedError

def test_logpdf_multirow_missing_values(exampleCGPM):
    raise NotImplementedError

def test_simulate_missing_values():
    raise NotImplementedError

## HELPERS ##

def analyze_until_view_partitions(model):
    # While there is only one view, analyze_until_view_partitions model
    i = 0
    while len(set(model.Zv().values())) == 1 and i < 10:
        model.transition(N=10)
        i += 1
    return model
