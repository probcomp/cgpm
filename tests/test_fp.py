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

import unittest

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
        regime = self._lookup(y)
        return gu.pflip(self.probabilities[regime], rng=self.rng)

    def logpdf(self, rowid, x, y):
        if not (0 <= x <= 3): return -float('inf')
        regime = self._lookup(y)
        return np.log(self.probabilities[regime][x])

    def _lookup(self, y):
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


class Dummy(object):
    def __init__(self): pass


class ForeignPredictorInferenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = gu.gen_rng(1)
        rows = 120
        cctypes = ['normal', 'bernoulli', 'normal']
        G = generate_quadrants(rows, rng)
        B, Zv, Zrv = tu.gen_data_table(
            rows, [1], [[.5,.5]], ['bernoulli'], [None], [.95], rng=rng)
        T = np.column_stack((G, B.T))[:,[0,2,1]]
        cls.state = State(T, cctypes)
        cls.state.transition(N=50)

        cls.four_way = FourWayPredictor(rng)
        cls.two_way = TwoWayPredictor(rng)

    def test_no_chained_predictors(self):
        four_index = self.state.update_foreign_predictor(self.four_way, [0, 2])

        with self.assertRaises(ValueError):
            self.state.update_foreign_predictor(
                self.two_way, [four_index, 1, 0])

        self.state.remove_foreign_predictor(four_index)

    def test_no_predictors_no_change(self):
        # Get some logpdfs and samples before predictors.
        logp_before_one = self.state.logpdf(2, [(0, 1), (1, 1)])
        logp_before_two = self.state.logpdf(-1, [(0, 1), (1, 1)], [(2,1)])
        simulate_before_one = self.state.simulate(2, [0,1,2], N=10)
        simulate_before_two = self.state.simulate(-1, [1,2], [(0,1)])

        # Hook the predictors.
        four_index = self.state.update_foreign_predictor(
            self.four_way, [0, 2])
        two_index = self.state.update_foreign_predictor(
            self.two_way, [1, 0])

        # Should be identical logpdfs and similar samples after.
        logp_after_one = self.state.logpdf(2, [(0, 1), (1, 1)])
        logp_after_two = self.state.logpdf(-1, [(0, 1), (1, 1)], [(2,1)])
        simulate_after_one = self.state.simulate(2, [0,1,2], N=10)
        simulate_after_two = self.state.simulate(-1, [1,2], [(0,1)])

        self.assertAlmostEqual(logp_before_one, logp_after_one)
        self.assertAlmostEqual(logp_before_two, logp_after_two)

        # Unhook the predictors.
        self.state.remove_foreign_predictor(four_index)
        self.state.remove_foreign_predictor(two_index)

    def crash_test_simulate_logpdf(self):
        # Incorporate the predictors.
        four_index = self.state.update_foreign_predictor(self.four_way, [0, 2])
        two_index = self.state.update_foreign_predictor(self.two_way, [1, 0])

        self.state.simulate(
            1, [0, 1, 2, four_index, two_index], N=100)
        self.state.simulate(
            -1, [0, 1, 2, four_index, two_index], N=100)

        self.state.logpdf(
            2, [(0,1), (1,0), (2,-1), (four_index, 3), (two_index, 0)])
        self.state.logpdf(
            -1, [(0,1), (1,0), (2,-1), (four_index, 3), (two_index, 0)])

        # Unhook the predictors.
        self.state.remove_foreign_predictor(four_index)
        self.state.remove_foreign_predictor(two_index)

    def test_inference_quality(self):
        # Incorporate the predictors.
        four_index = self.state.update_foreign_predictor(self.four_way, [0, 2])
        two_index = self.state.update_foreign_predictor(self.two_way, [1, 0])

        # Simulate parents (0, 2) constraining four_index.
        for v in [0, 1, 2, 3]:
            simulate_fourway_constrain = np.asarray(self.state.simulate(
                -1, [0, 2], [(four_index, v)], N=100))
            plt.figure()
            plt.scatter(
                simulate_fourway_constrain[:,0],
                simulate_fourway_constrain[:,1])

        # Simulate four_index varying parent constraints.
        for v in [0, 1, 2, 3]:
            x0, x1 = FourWayPredictor.retrieve_y_for_x(v)

            lp_exact = self.four_way.logpdf(None, v, [x0, x1])
            lp_fully_conditioned = self.state.logpdf(
                -1, [(four_index, v)],
                [(0, x0), (1, 1), (2, x1), (two_index, 0)])
            lp_missing_one = self.state.logpdf(
                -1, [(four_index, v)], [(0, x0), (1, 1)])
            lp_missing_two = self.state.logpdf(
                -1, [(four_index, v)])

            self.assertAlmostEqual(lp_fully_conditioned, lp_exact)
            self.assertLess(lp_missing_one, lp_fully_conditioned)
            self.assertLess(lp_missing_two, lp_missing_one)
            self.assertLess(lp_missing_two, lp_fully_conditioned)

            # Invert the query conditioning on four_index.
            lp_inverse_evidence = self.state.logpdf(
                -1, [(0, x0), (2, x1)], [(four_index, v)])
            lp_inverse_no_evidence = self.state.logpdf(
                -1, [(0, x0), (2, x1)])

            self.assertLess(lp_inverse_no_evidence, lp_inverse_evidence)

        # Simulate two_index varying parent constraints.
        for v in [0, 1]:
            x0, x1 = TwoWayPredictor.retrieve_y_for_x(v)

            lp_exact = self.two_way.logpdf(None, v, [x0, x1])
            lp_fully_conditioned = self.state.logpdf(
                -1, [(two_index, v)], [(1, x0), (0, x1), (2, x1)])
            lp_missing_one = self.state.logpdf(
                -1, [(two_index, v)], [(0, x1), (2, x1)])

            self.assertAlmostEqual(lp_fully_conditioned, lp_exact)
            self.assertLess(lp_missing_one, lp_fully_conditioned)

            # Invert the query conditioning on two_index.
            lp_inverse_evidence = self.state.logpdf(
                -1, [(1, x0), (0, x1)], [(two_index, v)])
            lp_inverse_no_evidence = self.state.logpdf(
                -1, [(1, x0), (0, x1)])

            self.assertLess(lp_inverse_no_evidence, lp_inverse_evidence)

        # Unhook the predictors.
        self.state.remove_foreign_predictor(four_index)
        self.state.remove_foreign_predictor(two_index)


if __name__ == '__main__':
    unittest.main()

