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


class FourWay(CGpm):
    """Outputs categorical(4) (quadrant indicator) on R2 valued input."""

    def __init__(self, outputs, inputs, rng):
        self.rng = rng
        self.probabilities =[
            [.7, .1, .05, .05],
            [.1, .8, .1, .1],
            [.1, .15, .65, .1],
            [.1, .05, .1, .75]]
        assert len(outputs) == 1
        assert len(inputs) == 2
        self.outputs = list(outputs)
        self.inputs = list(inputs)

    def incorporate(self, rowid, query, evidence=None):
        return

    def unincorporate(self, rowid):
        return

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        regime = self.lookup_quadrant(
            evidence[self.inputs[0]], evidence[self.inputs[1]])
        x = gu.pflip(self.probabilities[regime], rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf(self, rowid, query, evidence=None):
        x = query[self.outputs[0]]
        if not (0 <= x <= 3): return -float('inf')
        regime = self.lookup_quadrant(
            evidence[self.inputs[0]], evidence[self.inputs[1]])
        return np.log(self.probabilities[regime][x])

    def transition(self, N=None, S=None):
        time.sleep(.1)

    @staticmethod
    def lookup_quadrant(y0, y1):
        if y0 > 0 and y1 > 0: return 0
        if y0 < 0 and y1 > 0: return 1
        if y0 > 0 and y1 < 0: return 2
        if y0 < 0 and y1 < 0: return 3
        raise ValueError('Invalid value: %s' % str((y0, y1)))

    @staticmethod
    def retrieve_y_for_x(x):
        if x == 0: return [2, 2]
        if x == 1: return [-2, 2]
        if x == 2: return [2, -2]
        if x == 3: return [-2, -2]
        raise ValueError('Invalid value: %s' % str(x))

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['factory'] = ('cgpm.dummy.fourway', 'FourWay')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None: rng = gu.gen_rng(0)
        return cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            rng=rng)
