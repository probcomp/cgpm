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

import json
import tempfile
import unittest

import numpy as np

from gpmcc.engine import Engine
from gpmcc.state import State

from gpmcc.utils import general as gu


class TestSerialize(unittest.TestCase):

    def serialize_generic(self, Model, additional=None):
        """Model is either State or Engine class."""
        # Create categorical data of DATA_NUM_0 zeros and DATA_NUM_1 ones.
        data = np.random.normal(size=(100,5))
        data[:,0] = 0
        # Run a single chain for a few iterations.
        model = Model(
            data,
            cctypes=['bernoulli','normal','normal','normal','normal'],
            rng=gu.gen_rng(0))
        model.transition(N=1)
        # To metadata.
        metadata = model.to_metadata()
        Model.from_metadata(metadata)
        # To JSON.
        json_metadata = json.dumps(model.to_metadata())
        Model.from_metadata(json.loads(json_metadata))
        # To pickle.
        with tempfile.NamedTemporaryFile(prefix='gpmcc-serialize') as temp:
            with open(temp.name, 'w') as f:
                model.to_pickle(f)
            with open(temp.name, 'r') as f:
                model = Model.from_pickle(f, rng=gu.gen_rng(10))
                if additional:
                    additional(model)

    def test_state_serialize(self):
        self.serialize_generic(State)

    def test_engine_serialize(self):
        def additional(engine):
            e = engine.to_metadata()
            # Only one dataset per engine, not once per state.
            assert 'X' in e
            assert 'X' not in e['states'][0]
            # Each state should be populated with dataset when retrieving.
            s = engine.get_state(0)
            assert 'X' in s.to_metadata()
        self.serialize_generic(Engine, additional=additional)


if __name__ == '__main__':
    unittest.main()
