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

from gpmcc.gpm import Gpm
from gpmcc.uncorrelated.synthetic import SyntheticXyGpm
from gpmcc.uncorrelated.uniformx import UniformX
from gpmcc.utils.general import gen_rng


class DiamondY(Gpm):
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
        self.uniform = uniform(scale=self.noise)

    def simulate(self, rowid, query, evidence):
        assert query == self.outputs
        assert evidence.keys() == self.inputs
        x = evidence[self.inputs[0]]
        slope = self.rng.rand()
        noise = self.uniform.rvs(random_state=self.rng)
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
        return y

    def logpdf(self, rowid, query, evidence):
        raise NotImplementedError


class Diamond(SyntheticXyGpm):
    """Y = (+/- w.p .5) X^2 + U(0,noise)."""

    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        SyntheticXyGpm.__init__(
            self, outputs=outputs, inputs=inputs, noise=noise, rng=rng)
        self.x = UniformX(
            outputs=[self.outputs[0]], low=-1, high=1)
        self.y = DiamondY(
            outputs=[self.outputs[1]],
            inputs=[self.outputs[0]],
            noise=noise)
