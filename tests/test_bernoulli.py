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

import gpmcc.engine

DATA_NUM_0 = 100
DATA_NUM_1 = 200
NUM_SIM = 10000
NUM_ITER = 5

class TestBinomial(unittest.TestCase):

    def test_binomial(self):
        # Create categorical data of DATA_NUM_0 zeros and DATA_NUM_1 ones.
        data = np.transpose(np.array([[0] * DATA_NUM_0 + [1] * DATA_NUM_1]))
        # Run a single chain for a few iterations.
        engine = gpmcc.engine.Engine(
            data, ['categorical'], distargs=[{'k': 2}], seeds=[12],
            initialize=True)
        engine.transition(NUM_ITER)
        # Simulate from hypothetical row and compute the proportion of ones.
        xx = engine.simulate(-1, [0], N=NUM_SIM)[0]
        sum_b = np.sum(xx[:,0])
        observed_prob_of_1 = (float(sum_b) / float(NUM_SIM))
        true_prob_of_1 = float(DATA_NUM_1) / float(DATA_NUM_0 + DATA_NUM_1)
        # Check 1% relative match.
        assert np.allclose(true_prob_of_1, observed_prob_of_1, rtol=.1)

if __name__ == '__main__':
    unittest.main()
