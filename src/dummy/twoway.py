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

import time

import numpy as np

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class TwoWay(CGpm):
    """Generates {0,1} output on {0,1} valued input with given CPT."""

    def __init__(self, outputs, inputs, distargs=None, rng=None):
        if rng is None:
            rng = gu.gen_rng(1)
        self.rng = rng
        self.probabilities =[
            [.9, .1],
            [.3, .7],
        ]
        assert len(outputs) == 1
        assert len(inputs) == 1
        self.outputs = list(outputs)
        self.inputs = list(inputs)

    def incorporate(self, rowid, observation, inputs=None):
        return

    def unincorporate(self, rowid):
        return

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        y = inputs[self.inputs[0]]
        assert int(y) == float(y)
        assert y in [0, 1]
        x = gu.pflip(self.probabilities[y], rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        y = inputs[self.inputs[0]]
        assert int(y) == float(y)
        assert y in [0, 1]
        x = targets[self.outputs[0]]
        return np.log(self.probabilities[y][x]) if x in [0,1] else -float('inf')

    def transition(self, N=None):
        time.sleep(.1)

    @staticmethod
    def retrieve_y_for_x(x):
        if x == 0:
            return 0
        if x == 1:
            return 1
        raise ValueError('Invalid value: %s' % str(x))

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['factory'] = ('cgpm.dummy.twoway', 'TwoWay')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        return cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            rng=rng,
        )
