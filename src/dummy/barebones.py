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

"""A barebones cgpm only outputs, inputs, serialization, and not much else."""

from cgpm.cgpm import CGpm

class BareBonesCGpm(CGpm):

    def __init__(self, outputs, inputs, distargs=None, rng=None):
        self.outputs = outputs
        self.inputs = inputs

    def incorporate(self, rowid, observation, inputs=None):
        return

    def unincorporate(self, rowid):
        return

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        return 0

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        result = {i:1 for i in self.outputs}
        return result if N is None else [result] * N

    def transition(self, **kwargs):
        return

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['factory'] = ('cgpm.dummy.barebones', 'BareBonesCGpm')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        return cls(metadata['outputs'], metadata['inputs'])
