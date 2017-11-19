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


class TrollNormal(CGpm):
    def __init__(self, outputs, inputs, rng=None, distargs=None):
        if rng is None:
            rng = gu.gen_rng(1)
        self.rng = rng
        assert len(outputs) == 1
        assert len(inputs) == 2
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.rowids = set([])

    def incorporate(self, rowid, observation, inputs=None):
        assert rowid not in self.rowids
        self.rowids.add(rowid)

    def unincorporate(self, rowid):
        assert rowid in self.rowids
        self.rowids.remove(rowid)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        x = self.rng.normal(
            self._retrieve_location(inputs),
            self._retrieve_scale(inputs),
        )
        return {self.outputs[0]: x}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        return self._gaussian_log_pdf(
            targets[self.outputs[0]],
            self._retrieve_location(inputs),
            self._retrieve_scale(inputs),
        )

    def _gaussian_log_pdf(self, x, mu, s):
        normalizing_constant = -(np.log(2 * np.pi) / 2) - np.log(s)
        return normalizing_constant - ((x - mu)**2 / (2 * s**2))

    def transition(self, N=None, S=None):
        time.sleep(.1)

    def _retrieve_location(self, inputs):
        return inputs[self.inputs[0]]

    def _retrieve_scale(self, inputs):
        return abs(inputs[self.inputs[1]]) + 1

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['factory'] = ('cgpm.dummy.trollnormal', 'TrollNormal')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None: rng = gu.gen_rng(0)
        return cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            rng=rng)
