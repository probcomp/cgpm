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

'''This module tests conformance between direct and network-based
implementations of logpdf for cgpm.crosscat.State.'''

import numpy as np
import pytest

from cgpm.crosscat.engine import DummyCgpm
from cgpm.crosscat.engine import Engine
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


@pytest.fixture
def engine():
    # Set up the data generation
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'categorical(k=4)',
        'lognormal',
        'exponential',
        'beta',
        'geometric',
        'vonmises',
    ])

    T, Zv, Zc = tu.gen_data_table(
        20, [1], [[.25, .25, .5]], cctypes, distargs, [.95]*len(cctypes),
        rng=gu.gen_rng(10))

    return Engine(
        T.T,
        cctypes=cctypes,
        distargs=distargs,
        num_states=4,
        rng=gu.gen_rng(312),
        multiprocess=False
    )



def adhoc_state_logpdf_network(state, rowid, query, evidence):
    """Force state to use network logpdf by adding a cgpm."""
    dummy = DummyCgpm(inputs=state.outputs, outputs=[100000])
    token = state.compose_cgpm(dummy)
    assert state._composite
    logpdf = state.logpdf(rowid, query, evidence)
    state.decompose_cgpm(token)
    assert not state._composite
    return logpdf


def adhoc_state_simulate_network(state, rowid, query, evidence, N):
    """Force state to use network simulate by adding a cgpm."""
    dummy = DummyCgpm(inputs=state.outputs, outputs=[100000])
    token = state.compose_cgpm(dummy)
    assert state._composite
    samples = state.simulate(rowid, query, evidence, N)
    state.decompose_cgpm(token)
    assert not state._composite
    return samples


def test_logpdf_joint_conformance(engine):
    rowid = -1
    query = {0:1, 1:2, 2:1, 3:3, 4:1, 5:10, 6:.4, 7:2, 8:1.8}

    lp_direct = engine.logpdf(rowid, query)
    lp_network = [
        adhoc_state_logpdf_network(s, rowid, query, None)
        for s in engine.states
    ]
    assert np.allclose(lp_direct, lp_network)


def test_logpdf_conditional_conformance(engine):
    rowid = -1
    query = {1:2, 4:1, 5:10, 6:.4, 7:2, 8:1.8}
    evidence = {0:1, 2:1, 3:3}

    lp_direct = engine.logpdf(rowid, query, evidence)
    lp_network = [
        adhoc_state_logpdf_network(s, rowid, query, evidence)
        for s in engine.states
    ]
    assert np.allclose(lp_direct, lp_network)


def test_crash_logpdf_joint_observed(engine):
    rowid = 1
    query = {0:1, 1:2, 2:1, 3:3, 4:1, 5:10, 6:.4, 7:2, 8:1.8}

    for state in engine.states:
        with pytest.raises(ValueError):
            state.logpdf(rowid, query)
        with pytest.raises(ValueError):
            adhoc_state_logpdf_network(state, rowid, query, None)


def test_crash_logpdf_conditional_observed(engine):
    rowid = 1
    query = {0:1, 1:2, 2:1, 3:3, 4:1, 5:10, 6:.4, 7:2, 8:1.8}
    evidence = {0:1, 2:1, 3:3}

    for state in engine.states:
        with pytest.raises(ValueError):
            state.logpdf(rowid, query)
        with pytest.raises(ValueError):
            adhoc_state_logpdf_network(state, rowid, query, None)


def test_crash_simulate_conditional_observed(engine):
    rowid = 1
    query = [1, 4, 5, 6, 7, 8]
    evidence = {0:1, 2:1, 3:3}
    N = 10

    for state in engine.states:
        with pytest.raises(ValueError):
            state.simulate(rowid, query, evidence, N)
        with pytest.raises(ValueError):
            adhoc_state_simulate_network(state, rowid, query, evidence, N)


def test_logpdf_chain_rule(engine):
    from cgpm.utils.timer import Timer
    with Timer() as t:
        joint = engine.logpdf(-1, {0:1, 1:2}, multiprocess=False)
    print t.interval
    with Timer() as t:
        chain = np.add(
            engine.logpdf(-1, {0:1}, evidence={1:2}, multiprocess=False),
            engine.logpdf(-1, {1:2}, multiprocess=False))
    print t.interval
    assert np.allclose(joint, chain)

# if False:
#     state2 = State(T.T, cctypes=cctypes, distargs=distargs, rng=gu.gen_rng(12))
#     state2.transition(N=10, progress=1)

#     # Joint equals chain rule for state 2.
#     state2.logpdf(-1, {0:1, 1:2})
#     state2.logpdf(-1, {0:1}, evidence={1:2}) + state2.logpdf(-1, {1:2})

#     # Take the Monte Carlo average of the conditional.
#     mc_conditional = np.log(.5) + gu.logsumexp([
#         state.logpdf(-1, {0:1}, evidence={1:2}),
#         state2.logpdf(-1, {0:1}, evidence={1:2})
#     ])

#     # Take the Monte Carlo average of the joint.
#     mc_joint = np.log(.5) + gu.logsumexp([
#         state.logpdf(-1, {0:1, 1:2}),
#         state2.logpdf(-1, {0:1, 1:2})
#     ])

#     # Take the Monte Carlo average of the marginal.
#     mc_marginal = np.log(.5) + gu.logsumexp([
#         state.logpdf(-1, {1:2}),
#         state2.logpdf(-1, {1:2})
#     ])
