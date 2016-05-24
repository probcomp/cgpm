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

from gpmcc.state import State
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


class FourWayPredictor(object):
    """Foreign predictor on R2 valued input."""

    def __init__(self, rng):
        self.rng = rng
        self.probabilities =[
            [.7, .1, .05, .05],
            [.1, .8, .1, .1],
            [.1, .15, .65, .1],
            [.1, .05, .1, .75]]

    def simulate(self, rowid, y):
        regime = self.lookup_quadrant(y)
        return gu.pflip(self.probabilities[regime], rng=self.rng)

    def logpdf(self, rowid, x, y):
        if not (0 <= x <= 3): return -float('inf')
        regime = self.lookup_quadrant(y)
        return np.log(self.probabilities[regime][x])

    @staticmethod
    def lookup_quadrant(y):
        y0, y1 = y
        if y0 > 0 and y1 > 0: return 0
        if y0 < 0 and y1 > 0: return 1
        if y0 > 0 and y1 < 0: return 2
        if y0 < 0 and y1 < 0: return 3
        raise ValueError('Invalid value: %s' % str(y))

    @staticmethod
    def retrieve_y_for_x(x):
        if x == 0: return [2, 2]
        if x == 1: return [-2, 2]
        if x == 2: return [2, -2]
        if x == 3: return [-2, -2]
        raise ValueError('Invalid value: %s' % str(x))


class TwoWayPredictor(object):
    """Foreign predictor on binary valued input."""

    def __init__(self, rng):
        self.rng = rng
        self.probabilities =[
            [.9, .1],
            [.3, .7]]

    def simulate(self, rowid, y):
        assert int(y[0]) == float(y[0])
        y0 = int(y[0])
        return gu.pflip(self.probabilities[y0], rng=self.rng)

    def logpdf(self, rowid, x, y):
        assert int(y[0]) == float(y[0])
        x, y0 = int(x), int(y[0])
        if x not in [0, 1]: return -float('inf')
        return np.log(self.probabilities[y0][x])

    @staticmethod
    def retrieve_y_for_x(x):
        if x == 0: return [0, 0]
        if x == 1: return [1, 2]
        raise ValueError('Invalid value: %s' % str(x))


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


class Dummy(object):
    def __init__(self):
        pass


FOURWAY = FourWayPredictor(gu.gen_rng(1))
TWOWAY = TwoWayPredictor(gu.gen_rng(1))


@pytest.fixture(scope='module')
def state():
    rng = gu.gen_rng(1)
    rows = 120
    cctypes = ['normal', 'bernoulli', 'normal']
    G = generate_quadrants(rows, rng)
    B, Zv, Zrv = tu.gen_data_table(
        rows, [1], [[.5,.5]], ['bernoulli'], [None], [.95], rng=rng)
    T = np.column_stack((G, B.T))[:,[0,2,1]]
    state = State(T, cctypes)
    state.transition(N=50)
    return state


def test_no_chained_predictors(state):
    four_index = state.update_foreign_predictor(FOURWAY, [0, 2])

    with pytest.raises(ValueError):
        state.update_foreign_predictor(
            TWOWAY, [four_index, 1, 0])

    state.remove_foreign_predictor(four_index)


def test_no_predictors_no_change(state):
    # Get some logpdfs and samples before predictors.
    logp_before_one = state.logpdf(2, [(0, 1), (1, 1)])
    logp_before_two = state.logpdf(-1, [(0, 1), (1, 1)], [(2,1)])
    simulate_before_one = state.simulate(2, [0,1,2], N=10)
    simulate_before_two = state.simulate(-1, [1,2], [(0,1)])

    # Hook the predictors.
    four_index = state.update_foreign_predictor(
        FOURWAY, [0, 2])
    two_index = state.update_foreign_predictor(
        TWOWAY, [1, 0])

    # Should be identical logpdfs and similar samples after.
    logp_after_one = state.logpdf(2, [(0, 1), (1, 1)])
    logp_after_two = state.logpdf(-1, [(0, 1), (1, 1)], [(2,1)])
    simulate_after_one = state.simulate(2, [0,1,2], N=10)
    simulate_after_two = state.simulate(-1, [1,2], [(0,1)])

    assert np.allclose(logp_before_one, logp_after_one)
    assert np.allclose(logp_before_two, logp_after_two)

    # Unhook the predictors.
    state.remove_foreign_predictor(four_index)
    state.remove_foreign_predictor(two_index)


def crash_test_simulate_logpdf(state):
    # Incorporate the predictors.
    four_index = state.update_foreign_predictor(FOURWAY, [0, 2])
    two_index = state.update_foreign_predictor(TWOWAY, [1, 0])

    state.simulate(
        1, [0, 1, 2, four_index, two_index], N=100)
    state.simulate(
        -1, [0, 1, 2, four_index, two_index], N=100)

    state.logpdf(
        2, [(0,1), (1,0), (2,-1), (four_index, 3), (two_index, 0)])
    state.logpdf(
        -1, [(0,1), (1,0), (2,-1), (four_index, 3), (two_index, 0)])

    # Unhook the predictors.
    state.remove_foreign_predictor(four_index)
    state.remove_foreign_predictor(two_index)


def test_inference_quality(state):
    # Incorporate the predictors.
    four_index = state.update_foreign_predictor(FOURWAY, [0, 2])
    two_index = state.update_foreign_predictor(TWOWAY, [1, 0])

    # Simulate parents (0, 2) constraining four_index.
    for v in [0, 1, 2, 3]:
        simulate_fourway_constrain = np.asarray(state.simulate(
            -1, [0, 2], [(four_index, v)], N=100))
        plt.figure()
        plt.scatter(
            simulate_fourway_constrain[:,0],
            simulate_fourway_constrain[:,1])

        x0, x1 = FourWayPredictor.retrieve_y_for_x(v)
        simulate_ideal = np.asarray([[x0, x1]])

        counts_ideal = compute_quadrant_counts(simulate_ideal)
        counts_actual = compute_quadrant_counts(simulate_fourway_constrain)

        assert np.argmax(counts_ideal) == np.argmax(counts_actual)

    # logpdf four_index varying parent constraints.
    for v in [0, 1, 2, 3]:
        x0, x1 = FourWayPredictor.retrieve_y_for_x(v)

        lp_exact = FOURWAY.logpdf(None, v, [x0, x1])
        lp_fully_conditioned = state.logpdf(
            -1, [(four_index, v)],
            [(0, x0), (1, 1), (2, x1), (two_index, 0)])
        lp_missing_one = state.logpdf(
            -1, [(four_index, v)], [(0, x0), (1, 1)])
        lp_missing_two = state.logpdf(
            -1, [(four_index, v)])

        assert np.allclose(lp_fully_conditioned, lp_exact)
        assert lp_missing_one < lp_fully_conditioned
        assert lp_missing_two < lp_missing_one
        assert lp_missing_two < lp_fully_conditioned

        # Invert the query conditioning on four_index.
        lp_inverse_evidence = state.logpdf(
            -1, [(0, x0), (2, x1)], [(four_index, v)])
        lp_inverse_no_evidence = state.logpdf(
            -1, [(0, x0), (2, x1)])

        assert lp_inverse_no_evidence < lp_inverse_evidence

    # Simulate two_index varying parent constraints.
    for v in [0, 1]:
        x0, x1 = TwoWayPredictor.retrieve_y_for_x(v)

        lp_exact = TWOWAY.logpdf(None, v, [x0, x1])
        lp_fully_conditioned = state.logpdf(
            -1, [(two_index, v)], [(1, x0), (0, x1), (2, x1)])
        lp_missing_one = state.logpdf(
            -1, [(two_index, v)], [(0, x1), (2, x1)])

        assert np.allclose(lp_fully_conditioned, lp_exact)
        assert lp_missing_one < lp_fully_conditioned

        # Invert the query conditioning on two_index.
        lp_inverse_evidence = state.logpdf(
            -1, [(1, x0), (0, x1)], [(two_index, v)])
        lp_inverse_no_evidence = state.logpdf(
            -1, [(1, x0), (0, x1)])

        assert lp_inverse_no_evidence < lp_inverse_evidence

    # Unhook the predictors.
    state.remove_foreign_predictor(four_index)
    state.remove_foreign_predictor(two_index)
