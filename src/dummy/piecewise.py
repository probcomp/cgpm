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

import math
import time

import numpy as np

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class PieceWise(CGpm):

    def __init__(self, outputs, inputs, sigma=None, flip=None, distargs=None,
            rng=None):
        if rng is None:
            rng = gu.gen_rng(1)
        if sigma is None:
            sigma = 1
        if flip is None:
            flip = .5
        assert len(outputs) == 2
        assert len(inputs) == 1
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.sigma = sigma
        self.flip = flip
        self.rng = rng

    def incorporate(self, rowid, query, evidence=None):
        return

    def unincorporate(self, rowid):
        return

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        assert query
        assert self.inputs[0] in evidence
        y = evidence[self.inputs[0]]
        # Case 1: No evidence on outputs.
        if evidence.keys() == self.inputs:
            z = self.rng.choice([0, 1], p=[self.flip, 1-self.flip])
            x = y + (2*z-1) + self.rng.normal(0, self.sigma)
            sample = {}
            if self.outputs[0] in query:
                sample[self.outputs[0]] = x
            if self.outputs[1] in query:
                sample[self.outputs[1]] = z
        # Case 2: Simulating data given the latent.
        elif self.outputs[1] in evidence:
            assert query == [self.outputs[0]]
            z = evidence[self.outputs[1]]
            x = y + (2*z-1) + self.rng.normal(0, self.sigma)
            sample = {self.outputs[0]: x}
        # Case 3: Simulating latent given data.
        elif self.outputs[0] in evidence:
            assert query == [self.outputs[1]]
            # Compute probabilities for z | x,y
            p_z0 = self.logpdf(rowid, {self.outputs[1]: 0}, evidence)
            p_z1 = self.logpdf(rowid, {self.outputs[1]: 1}, evidence)
            z = self.rng.choice([0, 1], p=[np.exp(p_z0), np.exp(p_z1)])
            sample = {self.outputs[1]: z}
        else:
            raise ValueError('Misunderstood query: %s' % query)
        assert sample.keys() == query
        return sample

    def logpdf(self, rowid, query, evidence=None):
        assert query
        assert self.inputs[0] in evidence
        y = evidence[self.inputs[0]]
        # Case 1: No evidence on outputs.
        if evidence.keys() == self.inputs:
            # Case 1.1: z in the query and x in the query.
            if self.outputs[0] in query and self.outputs[1] in query:
                z, x = query[self.outputs[1]], query[self.outputs[1]]
                # XXX Check if z in [0, 1]
                logp_z = np.log(self.flip) if z == 0 else np.log(1-self.flip)
                logp_x = logpdf_normal(x, y + (2*z-1), self.sigma)
                logp = logp_x + logp_z
            # Case 1.2: z in the query only.
            elif self.outputs[1] in query:
                z = query[self.outputs[1]]
                logp_z = np.log(self.flip) if z == 0 else np.log(1-self.flip)
                logp = logp_z
            # Case 1.2: x in the query only.
            elif self.outputs[0] in query:
                x = query[self.outputs[0]]
                logp_xz0 = self.logpdf(
                    rowid, {self.outputs[0]: x, self.outputs[1]: 0}, evidence)
                logp_xz1 = self.logpdf(
                    rowid, {self.outputs[0]: x, self.outputs[1]: 1}, evidence)
                logp = gu.logsumexp([logp_xz0, logp_xz1])
            else:
                raise ValueError('Misunderstood query: %s.' % query)
        # Case 2: logpdf of x given the z.
        elif self.outputs[1] in evidence:
            assert query.keys() == [self.outputs[0]]
            z = evidence[self.outputs[1]]
            x = query[self.outputs[0]]
            logp_xz = self.logpdf(
                rowid, {self.outputs[0]: x, self.outputs[1]: z},
                {self.inputs[0]: y})
            logp_z = self.logpdf(rowid, {self.outputs[1]: z},
                {self.inputs[0]: y})
            logp = logp_xz - logp_z
        # Case 2: logpdf of z given the x.
        elif self.outputs[0] in evidence:
            assert query.keys() == [self.outputs[1]]
            z = query[self.outputs[1]]
            x = evidence[self.outputs[0]]
            logp_xz = self.logpdf(
                rowid, {self.outputs[0]: x, self.outputs[1]: z},
                {self.inputs[0]: y})
            logp_x = self.logpdf(rowid, {self.outputs[0]: x},
                {self.inputs[0]: y})
            logp = logp_xz - logp_x
        else:
            raise ValueError('Misunderstood query: %s.' % query)
        return logp

    def transition(self, N=None, S=None):
        time.sleep(.1)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['sigma'] = self.sigma
        metadata['flip'] = self.flip
        metadata['factory'] = ('cgpm.dummy.piecewise', 'PieceWise')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        return cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            sigma=metadata['sigma'],
            flip=metadata['flip'],
            rng=rng)


def logpdf_normal(x, mu, sigma):
    HALF_LOG2PI = 0.5 * math.log(2 * math.pi)
    deviation = x - mu
    return - math.log(sigma) - HALF_LOG2PI \
        - (0.5 * deviation * deviation / (sigma * sigma))
