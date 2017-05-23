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

import time

from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu
from cgpm.utils import test as tu

from markers import integration


def retrieve_normal_dataset():
    D, Zv, Zc = tu.gen_data_table(
        n_rows=20,
        view_weights=None,
        cluster_weights=[[.2,.2,.2,.4],[.2,.8],],
        cctypes=['normal', 'normal'],
        distargs=[None]*2,
        separation=[0.95]*2,
        view_partition=[0,1],
        rng=gu.gen_rng(12))
    return D


@integration
def test_simple_diagnostics():
    def diagnostics_without_iters(diagnostics):
        return (v for k, v in diagnostics.iteritems() if k != 'iterations')
    D = retrieve_normal_dataset()
    engine = Engine(
            D.T, cctypes=['normal']*len(D),  num_states=4, rng=gu.gen_rng(12),)
    engine.transition(N=20, checkpoint=2)
    assert all(
        all(len(v) == 10 for v in diagnostics_without_iters(state.diagnostics))
        for state in engine.states
    )
    engine.transition(N=7, checkpoint=2)
    assert all(
        all(len(v) == 13 for v in diagnostics_without_iters(state.diagnostics))
        for state in engine.states
    )
    engine.transition_lovecat(N=7, checkpoint=3)
    assert all(
        all(len(v) == 15 for v in diagnostics_without_iters(state.diagnostics))
        for state in engine.states
    )
    engine.transition(S=0.5)
    assert all(
        all(len(v) == 15 for v in diagnostics_without_iters(state.diagnostics))
        for state in engine.states
    )
    engine.transition(S=0.5, checkpoint=1)
    assert all(
        all(len(v) > 15 for v in diagnostics_without_iters(state.diagnostics))
        for state in engine.states
    )
    # Add a timed analysis with diagnostic overrides large iterations, due
    # to oddness of diagnostic tracing in lovecat.
    start = time.time()
    engine.transition_lovecat(N=20000, S=1, checkpoint=1)
    assert 1 < time.time() - start < 3
