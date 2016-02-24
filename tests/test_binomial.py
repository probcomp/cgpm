# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2016 MIT Probabilistic Computing Project

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

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
        # Check that the observed proportion of ones is about 2/3, out to two
        # places. If this were a plain binomial model, we'd expect the sample
        # proportion to have standard deviation sqrt(2/3*1/3/10000) = 0.005, so
        # check out to two places.
        self.assertAlmostEqual(true_prob_of_1, observed_prob_of_1, places=1)

if __name__ == '__main__':
    unittest.main()
