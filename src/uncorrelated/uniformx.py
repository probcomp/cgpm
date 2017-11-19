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

from scipy.stats import uniform

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class UniformX(CGpm):

    def __init__(self, outputs=None, inputs=None, low=0, high=1, rng=None):
        assert not inputs
        if rng is None:
            rng = gu.gen_rng(0)
        if outputs is None:
            outputs = [0]
        self.rng = rng
        self.low = low
        self.high = high
        self.outputs = outputs
        self.inputs = []
        self.uniform = uniform(loc=self.low, scale=self.high-self.low)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert not constraints
        assert targets == self.outputs
        x = self.rng.uniform(low=self.low, high=self.high)
        return {self.outputs[0]: x}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert not constraints
        assert not inputs
        assert targets.keys() == self.outputs
        x = targets[self.outputs[0]]
        return self.uniform.logpdf(x)
