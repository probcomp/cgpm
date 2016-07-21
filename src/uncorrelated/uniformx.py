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
from cgpm.utils.general import gen_rng


class UniformX(CGpm):

    def __init__(self, outputs=None, inputs=None, low=0, high=1, rng=None):
        assert not inputs
        if rng is None:
            rng = gen_rng(0)
        if outputs is None:
            outputs = [0]
        self.rng = rng
        self.low = low
        self.high = high
        self.outputs = outputs
        self.inputs = []
        self.uniform = uniform(loc=self.low, scale=self.high-self.low)

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        assert not evidence
        assert query == self.outputs
        x = self.rng.uniform(low=self.low, high=self.high)
        return {self.outputs[0]: x}

    def logpdf(self, rowid, query, evidence=None):
        assert not evidence
        assert query.keys() == self.outputs
        x = query[self.outputs[0]]
        return self.uniform.logpdf(x)
