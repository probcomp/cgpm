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

import numpy as np

from scipy.misc import logsumexp
from scipy.stats import uniform

from gpmcc.cgpm import CGpm
from gpmcc.network.importance import ImportanceNetwork
from gpmcc.uncorrelated.uniformx import UniformX
from gpmcc.uncorrelated.directed import DirectedXyGpm
from gpmcc.utils.general import gen_rng


class ParabolaY(CGpm):
    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        if rng is None:
            rng = gen_rng(1)
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
        self.uniform = uniform(loc=-self.noise, scale=2*self.noise)

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        assert query == self.outputs
        assert evidence.keys() == self.inputs
        x = evidence[self.inputs[0]]
        u = self.rng.rand()
        noise = self.uniform.rvs(random_state=self.rng)
        if u < .5:
            y = x**2 + noise
        else:
            y = -(x**2 + noise)
        return {self.outputs[0]: y}

    def logpdf(self, rowid, query, evidence=None):
        assert query.keys() == self.outputs
        assert evidence.keys() == self.inputs
        x = evidence[self.inputs[0]]
        y = query[self.outputs[0]]
        return logsumexp([
            np.log(.5)+self.uniform.logpdf(y-x**2),
            np.log(.5)+self.uniform.logpdf(-y-x**2)])


class Parabola(DirectedXyGpm):
    """Y = (+/- w.p .5) X^2 + U(0,noise)."""

    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        DirectedXyGpm.__init__(
            self, outputs=outputs, inputs=inputs, noise=noise, rng=rng)
        self.x = UniformX(
            outputs=[self.outputs[0]], low=-1, high=1)
        self.y = ParabolaY(
            outputs=[self.outputs[1]],
            inputs=[self.outputs[0]],
            noise=noise)
        self.network = ImportanceNetwork([self.x, self.y], rng=self.rng)
