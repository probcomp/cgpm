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

import numpy as np
import pytest

from cgpm.crosscat.engine import Engine
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


# Set up the data generation
def get_engine():
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'lognormal',
        'beta',
        'vonmises'
    ])
    T, Zv, Zc = tu.gen_data_table(
        20, [1], [[.25, .25, .5]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(0))
    T = T.T
    # Make some nan cells for evidence.
    T[5,0] = T[5,1] = T[5,2] = T[5,3] = np.nan
    T[8,4] = np.nan
    engine = Engine(
        T,
        cctypes=cctypes,
        distargs=distargs,
        num_states=6,
        rng=gu.gen_rng(0)
    )
    engine.transition(N=2)
    return engine


def test_simple_alterations():
    engine = get_engine()

    # Initial state outputs.
    out_initial = engine.states[0].outputs

    # Indexes of outputs to alter.
    out_f = 0
    out_g = 3

    def alteration_f(state):
        state.outputs[out_f] *= 13
        return state

    def alteration_g(state):
        state.outputs[out_g] *= 12
        return state

    statenos = [0,3]
    engine.alter((alteration_f, alteration_g), [0,3])

    out_expected = list(out_initial)
    out_expected[out_f] *= 13
    out_expected[out_g] *= 12

    for s in xrange(engine.num_states()):
        if s in statenos:
            assert engine.states[s].outputs == out_expected
        else:
            assert engine.states[s].outputs == out_initial
