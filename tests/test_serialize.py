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
            num_states=4)
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
