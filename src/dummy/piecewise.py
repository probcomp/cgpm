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
    """Generates data from a linear model with a 2-mixture over the slope.

    y := exogenous covariate
    z ~ Bernoulli(.5)
    x | z; y = y + (2*z - 1) + Normal(0, \sigma)
    """

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

    def incorporate(self, rowid, observation, inputs=None):
        return

    def unincorporate(self, rowid):
        return

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets
        assert inputs.keys() == self.inputs
        y = inputs[self.inputs[0]]
        # Case 1: No constraints on outputs.
        if not constraints:
            z = self.rng.choice([0, 1], p=[self.flip, 1-self.flip])
            x = y + (2*z-1) + self.rng.normal(0, self.sigma)
            sample = {}
            if self.outputs[0] in targets:
                sample[self.outputs[0]] = x
            if self.outputs[1] in targets:
                sample[self.outputs[1]] = z
        # Case 2: Simulating x given the z.
        elif constraints.keys() == [self.outputs[1]]:
            assert targets == [self.outputs[0]]
            z = constraints[self.outputs[1]]
            x = y + (2*z - 1) + self.rng.normal(0, self.sigma)
            sample = {self.outputs[0]: x}
        # Case 3: Simulating z given the x.
        elif constraints.keys() == [self.outputs[0]]:
            assert targets == [self.outputs[1]]
            # Compute probabilities for z | x,y
            p_z0 = self.logpdf(rowid, {self.outputs[1]: 0}, constraints, inputs)
            p_z1 = self.logpdf(rowid, {self.outputs[1]: 1}, constraints, inputs)
            z = self.rng.choice([0, 1], p=[np.exp(p_z0), np.exp(p_z1)])
            sample = {self.outputs[1]: z}
        else:
            raise ValueError('Invalid query pattern: %s, %s, %s'
                % (targets, constraints, inputs))
        assert sorted(sample.keys()) == sorted(targets)
        return sample

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert targets
        assert inputs.keys() == self.inputs
        y = inputs[self.inputs[0]]
        # Case 1: No evidence on outputs.
        if not constraints:
            # Case 1.1: z in the targets and x in the targets.
            if self.outputs[0] in targets and self.outputs[1] in targets:
                z, x = targets[self.outputs[1]], targets[self.outputs[1]]
                # XXX Check if z in [0, 1]
                logp_z = np.log(self.flip) if z == 0 else np.log(1-self.flip)
                logp_x = logpdf_normal(x, y + (2*z - 1), self.sigma)
                logp = logp_x + logp_z
            # Case 1.2: z in the targets only.
            elif self.outputs[1] in targets:
                z = targets[self.outputs[1]]
                logp_z = np.log(self.flip) if z == 0 else np.log(1-self.flip)
                logp = logp_z
            # Case 1.2: x in the targets only.
            elif self.outputs[0] in targets:
                x = targets[self.outputs[0]]
                logp_xz0 = self.logpdf(
                    rowid,
                    {self.outputs[0]: x, self.outputs[1]: 0},
                    constraints,
                    inputs
                )
                logp_xz1 = self.logpdf(
                    rowid,
                    {self.outputs[0]: x, self.outputs[1]: 1},
                    constraints,
                    inputs,
                )
                logp = gu.logsumexp([logp_xz0, logp_xz1])
            else:
                raise ValueError('Invalid query pattern: %s %s %s'
                    % (targets, constraints, inputs))
        # Case 2: logpdf of x given the z.
        elif constraints.keys() == [self.outputs[1]]:
            assert targets.keys() == [self.outputs[0]]
            z = constraints[self.outputs[1]]
            x = targets[self.outputs[0]]
            logp_xz = self.logpdf(
                rowid,
                {self.outputs[0]: x, self.outputs[1]: z},
                None,
                {self.inputs[0]: y}
            )
            logp_z = self.logpdf(
                rowid,
                {self.outputs[1]: z},
                None,
                {self.inputs[0]: y}
            )
            logp = logp_xz - logp_z
        # Case 2: logpdf of z given the x.
        elif constraints.keys() == [self.outputs[0]]:
            assert targets.keys() == [self.outputs[1]]
            z = targets[self.outputs[1]]
            x = constraints[self.outputs[0]]
            logp_xz = self.logpdf(
                rowid,
                {self.outputs[0]: x, self.outputs[1]: z},
                None,
                {self.inputs[0]: y}
            )
            logp_x = self.logpdf(
                rowid,
                {self.outputs[0]: x},
                None,
                {self.inputs[0]: y}
            )
            logp = logp_xz - logp_x
        else:
            raise ValueError('Invalid query pattern: %s %s %s'
                % (targets, constraints, inputs))
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
