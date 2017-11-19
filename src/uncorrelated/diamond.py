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

from cgpm.cgpm import CGpm
from cgpm.network.importance import ImportanceNetwork
from cgpm.uncorrelated.directed import DirectedXyGpm
from cgpm.uncorrelated.uniformx import UniformX
from cgpm.utils import general as gu


class DiamondY(CGpm):
    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        if rng is None:
            rng = gu.gen_rng(1)
        if outputs is None:
            outputs = [0]
        if inputs is None:
            inputs = [1]
        if noise is None:
            noise = .1
        self.rng = rng
        self.outputs = outputs
        self.inputs = inputs
        self.noise = noise

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets == self.outputs
        assert inputs.keys() == self.inputs
        x = inputs[self.inputs[0]]
        slope = self.rng.rand()
        noise = self.rng.uniform(high=self.noise)
        if x < 0 and slope < .5:
            y = max(-x-1, x+1 - noise)
        elif x < 0 and slope > .5:
            y = min(x+1, -x-1 + noise)
        elif x > 0 and slope < .5:
            y = min(-x+1, x-1 + noise)
        elif x > 0 and slope > .5:
            y = max(x-1, -x+1 - noise)
        else:
            raise ValueError()
        return {self.outputs[0]: y}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        raise NotImplementedError


class Diamond(DirectedXyGpm):
    """Y = (+/- w.p .5) X^2 + U(0,noise)."""

    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        DirectedXyGpm.__init__(
            self, outputs=outputs, inputs=inputs, noise=noise, rng=rng)
        self.x = UniformX(
            outputs=[self.outputs[0]], low=-1, high=1)
        self.y = DiamondY(
            outputs=[self.outputs[1]],
            inputs=[self.outputs[0]],
            noise=noise)
        self.network = ImportanceNetwork([self.x, self.y], rng=self.rng)
