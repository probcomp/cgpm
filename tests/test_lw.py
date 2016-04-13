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

import unittest

from gpmcc.state import State
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu

"""Crash and sanity tests for queries using likelihood weighting inference with
a RandomForest component model. Not an inference quality test suite."""

class LikelihoodWeightSanityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cctypes, distargs = cu.parse_distargs([
            'categorical(k=5)',
            'normal',
            'poisson',
            'bernoulli'])
        T, Zv, Zc = tu.gen_data_table(50, [1], [[.33, .33, .34]], cctypes,
            distargs, [.95]*len(cctypes), rng=gu.gen_rng(0))
        state = State(T.T, cctypes, distargs=distargs, Zv=[0]*len(cctypes),
            rng=gu.gen_rng(0))
        state.update_cctype(0, 'random_forest', distargs={'k':5})
        state.transition(N=10, kernels=['rows','view_alphas','alpha',
            'column_params','column_hypers'])
        cls.state = state

    def test_simulate_unconditional__ci_(self):
        for rowid in [-1, 1]:
            samples = self.state.simulate(rowid, [0], N=2)
            self.check_members(samples, range(5))

    def test_simulate_conditional__ci_(self):
        samples = self.state.simulate(
            -1, [0], evidence=[(1,-1), (2,1), (3,1)], N=2)
        self.check_members(samples, range(5))
        samples = self.state.simulate(-1, [0, 2, 3], N=2)
        self.check_members(samples, range(5))
        samples = self.state.simulate(1, [0, 2, 3], N=2)
        self.check_members(samples, range(5))

    def test_logpdf_unconditional__ci_(self):
        for rowid, k in zip([-1, 1], xrange(5)):
            self.assertLess(self.state.logpdf(rowid, [(0, k)]), 0)

    def test_logpdf_deterministic__ci_(self):
        # Ensure logpdf estimation deterministic when all parents in evidence.
        for k in xrange(5):
            lp1 = self.state.logpdf(
                -1, [(0,k), (3,0)], evidence=[(1,1), (2,1),])
            lp2 = self.state.logpdf(
                -1, [(0,k), (3,0)], evidence=[(1,1), (2,1),])
            self.assertAlmostEqual(lp1, lp2)
        # Observed cell already has parents in evidence.
        for k in xrange(5):
            lp1 = self.state.logpdf(1, [(0,k), (3,0)])
            lp2 = self.state.logpdf(1, [(0,k), (3,0)])
            self.assertAlmostEqual(lp1, lp2)

    def check_members(self, samples, allowed):
        for s in samples:
            self.assertIn(s[0], allowed)

if __name__ == '__main__':
    unittest.main()
