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

"""Inference quality tests for the conditional GPM  (aka foreign predictor)
features of State."""

import pytest

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

from cgpm.cgpm import CGpm
from cgpm.crosscat.state import State
from cgpm.dummy.fourway import FourWay
from cgpm.dummy.twoway import TwoWay
from cgpm.utils import general as gu
from cgpm.utils import test as tu


def generate_quadrants(rows, rng):
    Q0 = rng.multivariate_normal([2,2], cov=[[.5,0],[0,.5]], size=rows/4)
    Q1 = rng.multivariate_normal([-2,2], cov=[[.5,0],[0,.5]], size=rows/4)
    Q2 = rng.multivariate_normal([-2,-2], cov=[[.5,0],[0,.5]], size=rows/4)
    Q3 = rng.multivariate_normal([2,-2], cov=[[.5,0],[0,.5]], size=rows/4)
    colors = iter(cm.gist_rainbow(np.linspace(0, 1, 4)))
    for q in [Q0, Q1, Q2, Q3]:
        plt.scatter(q[:,0], q[:,1], color=next(colors))
    plt.close('all')
    return np.row_stack((Q0, Q1, Q2, Q3))


def compute_quadrant_counts(T):
    c0 = sum(np.logical_and(T[:,0] > 0, T[:,1] > 0))
    c1 = sum(np.logical_and(T[:,0] < 0, T[:,1] > 0))
    c2 = sum(np.logical_and(T[:,0] > 0, T[:,1] < 0))
    c3 = sum(np.logical_and(T[:,0] < 0, T[:,1] < 0))
    return [c0, c1, c2, c3]


@pytest.fixture(scope='module')
def state():
    rng = gu.gen_rng(5)
    rows = 120
    cctypes = ['normal', 'bernoulli', 'normal']
    G = generate_quadrants(rows, rng)
    B, Zv, Zrv = tu.gen_data_table(
        rows, [1], [[.5,.5]], ['bernoulli'], [None], [.95], rng=rng)
    T = np.column_stack((G, B.T))[:,[0,2,1]]
    state = State(T, outputs=[0,1,2], cctypes=cctypes, rng=rng)
    state.transition(N=20)
    return state


def test_duplicated_outputs(state):
    """This test ensures that foreign cgpms cannot collide on outputs."""
    for o in state.outputs:
        fourway = FourWay([o], [0,2], rng=state.rng)
        with pytest.raises(ValueError):
            state.compose_cgpm(fourway)
            assert len(state.hooked_cgpms) == 0
            assert not state._composite


def test_decompose_cgpm(state):
    """This test ensures that foreign cgpms can be composed and decomposing
    using the returned tokens."""
    four = FourWay([15], [0,2], rng=state.rng)
    two = TwoWay([10], [1], rng=state.rng)
    four_token = state.compose_cgpm(four)
    two_token = state.compose_cgpm(two)
    assert state.hooked_cgpms[four_token] == four
    assert state.hooked_cgpms[two_token] == two
    assert state._composite
    state.decompose_cgpm(two_token)
    assert state.hooked_cgpms[four_token] == four
    assert state._composite
    state.decompose_cgpm(four_token)
    assert len(state.hooked_cgpms) == 0
    assert not state._composite


def test_same_logpdf(state):
    """This test ensures that composing gpmcc with foreign cgpms does not change
    logpdf values for queries not involving the child cgpms."""

    # Get some logpdfs and samples before composing with cgpms.
    logp_before_one = state.logpdf(-1, {0: 1, 1: 1}, None, None)
    logp_before_two = state.logpdf(-1, {0: 1, 1: 1}, {2:1}, None)
    simulate_before_one = state.simulate(-1, [0,1,2], None, None, 10)
    simulate_before_two = state.simulate(-1, [1,2], {0:1}, None)

    # Compose the CGPMs.
    four_index = state.compose_cgpm(FourWay([5], [0,2], rng=state.rng))
    two_index = state.compose_cgpm(TwoWay([10], [1], rng=state.rng))

    # Get some logpdfs and samples after composing with cgpms.
    logp_after_one = state.logpdf(-1, {0: 1, 1: 1})
    logp_after_two = state.logpdf(-1, {0: 1, 1: 1}, {2:1})
    simulate_after_one = state.simulate(-1, [0,1,2], N=10)
    simulate_after_two = state.simulate(-1, [1,2], {0:1})

    # Check logps same.
    assert np.allclose(logp_before_one, logp_after_one)
    assert np.allclose(logp_before_two, logp_after_two)

    # Decompose the CGPMs.
    state.decompose_cgpm(four_index)
    state.decompose_cgpm(two_index)


def crash_test_simulate_logpdf(state):
    """This crash test ensures foreign cgpms can be composed and queried."""

    four_token = state.compose_cgpm(FourWay([5], [0,2], rng=state.rng))
    two_token = state.compose_cgpm(FourWay([5], [0,2], rng=state.rng))

    state.simulate(-1, [0, 1, 2, 5, 10], N=10)
    state.logpdf(-1, {0:1, 1:0, 2:-1, 5:3, 10:0})

    state.simulate(-1, [5, 0], {10:0, 2:-1}, N=10)
    state.logpdf(-1, {5:1, 0:2}, {10:0, 2:-1})

    # Unhook the predictors.
    state.decompose_cgpm(four_token)
    state.decompose_cgpm(two_token)


def test_inference_quality__ci_(state):
    """This test explores inference quality for simulate/logpdf inversion."""
    # Build CGPMs.
    fourway = FourWay([5], [0,2], rng=state.rng)
    twoway = TwoWay([10], [1], rng=state.rng)

    # Compose.
    four_token = state.compose_cgpm(fourway)
    two_token = state.compose_cgpm(twoway)

    # simulate parents (0,2) constraining four_index.
    for v in [0, 1, 2, 3]:
        samples = state.simulate(-1, [0, 2], {5: v}, N=100, accuracy=20)
        simulate_fourway_constrain = np.transpose(
            np.asarray([[s[0] for s in samples], [s[2] for s in samples]]))

        fig, ax = plt.subplots()
        ax.scatter(
            simulate_fourway_constrain[:,0],
            simulate_fourway_constrain[:,1])
        ax.hlines(0, *ax.get_xlim(),color='red', linewidth=2)
        ax.vlines(0, *ax.get_ylim(),color='red', linewidth=2)
        ax.grid()

        x0, x1 = FourWay.retrieve_y_for_x(v)
        simulate_ideal = np.asarray([[x0, x1]])
        counts_ideal = compute_quadrant_counts(simulate_ideal)
        counts_actual = compute_quadrant_counts(simulate_fourway_constrain)
        assert np.argmax(counts_ideal) == np.argmax(counts_actual)

    # logpdf four_index varying parent constraints.
    for v in [0, 1, 2, 3]:
        x0, x1 = FourWay.retrieve_y_for_x(v)

        lp_exact = fourway.logpdf(None, {5:v}, None, {0:x0, 2:x1})
        lp_fully_conditioned = state.logpdf(
            None, {5:v}, {0:x0, 1:1, 2:x1, 10:0}, accuracy=100)
        lp_missing_one = state.logpdf(None, {5:v}, {0: x0, 1:1}, accuracy=100)
        lp_missing_two = state.logpdf(None, {5:v}, accuracy=100)

        assert np.allclose(lp_fully_conditioned, lp_exact)
        assert lp_missing_one < lp_fully_conditioned
        assert lp_missing_two < lp_missing_one
        assert lp_missing_two < lp_fully_conditioned

        # Invert the query conditioning on four_index.
        lp_inverse_evidence = state.logpdf(
            -1, {0:x0, 2:x1}, {5: v}, accuracy=100)
        lp_inverse_no_evidence = state.logpdf(
            -1, {0:x0, 2: x1})

        assert lp_inverse_no_evidence < lp_inverse_evidence

    # Simulate two_index varying parent constraints.
    for v in [0, 1]:
        x1 = TwoWay.retrieve_y_for_x(v)

        lp_exact = twoway.logpdf(None, {10:v}, None, {0:0, 1:x1})
        lp_fully_conditioned = state.logpdf(None, {10: v}, {0:0, 1:x1, 2:1})
        lp_missing_one = state.logpdf(None, {10: v}, {0:0, 2:1}, accuracy=200)

        assert np.allclose(lp_fully_conditioned, lp_exact)
        assert lp_missing_one < lp_fully_conditioned

        # Invert the query conditioning on two_index.
        lp_inverse_evidence = state.logpdf(
            None, {0:0, 1:x1}, {10:v}, accuracy=100)
        lp_inverse_no_evidence = state.logpdf(None, {0:0, 1:x1})

        assert lp_inverse_no_evidence < lp_inverse_evidence

    # Unhook the predictors.
    state.decompose_cgpm(four_token)
    state.decompose_cgpm(two_token)
