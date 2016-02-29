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

import numpy as np

from gpmcc.state import State

class TestMutualInformation(unittest.TestCase):

    def __ci_test_continuous_discrete(self):
        # Test that continuous and discrete CMI is not allowed.
        rng = np.random.RandomState(0)
        T = np.zeros((50,2))
        T[:,0] = rng.normal(size=50)
        T[:,1] = rng.poisson(size=50)
        state = State(T, ['normal','poisson'])
        state.transition(N=1, do_progress=0)
        with self.assertRaises(ValueError):
            state.mutual_information(0,1)

    def __ci_test_numeric_continuous_symbolic(self):
        rng = np.random.RandomState(0)
        T = np.zeros((50,2))
        T[:,0] = rng.normal(size=50)
        T[:,1] = rng.choice(range(3), size=50)
        state = State(T, ['normal','categorical'], distargs=[None, {'k':3}])
        state.transition(N=1, do_progress=0)
        with self.assertRaises(ValueError):
            state.mutual_information(0,1)

    def __ci_test_discrete_numeric_symbolic(self):
        rng = np.random.RandomState(0)
        cctypes = ['poisson','categorical']
        T = np.zeros((50,2))
        T[:,0] = rng.poisson(size=50)
        T[:,1] = rng.choice(range(3), size=50)
        state = State(T, cctypes, distargs=[None, {'k':3}])
        state.transition(N=1, do_progress=0)
        with self.assertRaises(ValueError):
            state.mutual_information(0,1)

    def __ci_test_entropy_bernoulli(self):
        rng = np.random.RandomState(0)
        T = rng.choice([0,1], p=[.3,.7], size=1000).reshape(-1,1)
        state = State(T, ['bernoulli'], seed=0)
        state.transition(N=30, do_progress=0)
        # Exact computation.
        logp = state.logpdf_bulk([-1,-1],[[(0,0)],[(0,1)]])
        entropy_exact = -np.sum(np.exp(logp)*logp)
        # Monte Carlo computation.
        entropy_mc = state.mutual_information(0,0,N=1000)
        # Punt CLT analysis and go for 1 percent.
        assert np.allclose(entropy_exact, entropy_mc, rtol=0.1)

    def __ci_test_cmi_different_views(self):
        rng = np.random.RandomState(0)
        T = np.zeros((50,3))
        T[:,0] = rng.normal(loc=-5, scale=1, size=50)
        T[:,1] = rng.normal(loc=2, scale=2, size=50)
        T[:,2] = rng.normal(loc=12, scale=3, size=50)
        state = State(T, ['normal','normal','normal'], Zv=[0,1,2], seed=0)
        state.transition(N=30, kernels=['alpha', 'view_alphas',
            'column_params', 'column_hypers','rows'])

        mi01 = state.mutual_information(0,1)
        mi02 = state.mutual_information(0,2)
        mi12 = state.mutual_information(1,2)

        # Marginal MI all zero.
        self.assertAlmostEqual(mi01, 0)
        self.assertAlmostEqual(mi02, 0)
        self.assertAlmostEqual(mi12, 0)

        # CMI on variable in other view equal to MI.
        self.assertAlmostEqual(
            state.mutual_information(0, 1, evidence=[(2,10)]), mi01)
        self.assertAlmostEqual(
            state.mutual_information(0, 2, evidence=[(1,0)]), mi02)
        self.assertAlmostEqual(
            state.mutual_information(1, 2, evidence=[(0,-2)]), mi12)

if __name__ == '__main__':
    unittest.main()
