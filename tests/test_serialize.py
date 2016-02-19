# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

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

"""Crash test for serialization of state and engine."""

import unittest
import tempfile

import numpy as np

import gpmcc.state
import gpmcc.engine

class TestSerialize(unittest.TestCase):

    def test_state_serialize(self):
        # Create categorical data of DATA_NUM_0 zeros and DATA_NUM_1 ones.
        data = np.random.normal(size=(100,5))
        data[:,0] = 0
        # Run a single chain for a few iterations.
        state = gpmcc.state.State(
            data, ['bernoulli','normal','normal','normal','normal'])
        state.transition(N=1)
        # To JSON.
        metadata = state.to_metadata()
        state_clone = gpmcc.engine.State.from_metadata(metadata)
        # To pickle.
        with tempfile.NamedTemporaryFile(prefix='gpmcc-state') as temp:
            with open(temp.name, 'w') as f:
                state.to_pickle(f)
            with open(temp.name, 'r') as f:
                state_clone = state.from_pickle(f)

    def test_engine_serialize(self):
        # Create categorical data of DATA_NUM_0 zeros and DATA_NUM_1 ones.
        data = np.random.normal(size=(100,5))
        data[:,0] = 0
        # Run a single chain for a few iterations.
        engine = gpmcc.engine.Engine(
            data, ['bernoulli','normal','normal','normal','normal'],
            initialize=True, num_states=4)
        engine.transition(N=1)
        # To JSON.
        metadata = engine.to_metadata()
        engine_clone = gpmcc.engine.Engine.from_metadata(metadata)
        # To pickle.
        with tempfile.NamedTemporaryFile(prefix='gpmcc-engine') as temp:
            with open(temp.name, 'w') as f:
                engine.to_pickle(f)
            with open(temp.name, 'r') as f:
                engine_clone = engine.from_pickle(f)

if __name__ == '__main__':
    unittest.main()
