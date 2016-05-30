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

from scipy.stats import multivariate_normal
from scipy.stats import norm

from gpmcc.gpm import Gpm
from gpmcc.utils.general import gen_rng


class Linear(Gpm):
    """Y = X + N(0,noise), it is NOT uncorrelated."""

    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        if rng is None:
            rng = gen_rng(0)
        if outputs is None:
            outputs = [0, 1]
        if noise is None:
            noise = .1
        self.rng = rng
        self.outputs = outputs
        self.inputs = []
        self.noise = noise

    def simulate(self, rowid, query, evidence=None):
        if not evidence:
            sample = self.simulate_joint()
            return [sample[self.outputs.index(q)] for q in query]
        else:
            assert len(evidence) == len(query) == 1
            z = evidence.values()[0]
            return self.simulate_conditional(z)

    def logpdf(self, rowid, query, evidence):
        if not evidence:
            if len(query) == 2:
                x, y = query.values()
                return self.logpdf_joint(x, y)
            else:
                z = query.values()[0]
                return self.logpdf_maringal(z)
        else:
            assert len(evidence) == len(query) == 1
            z = evidence.values()[0]
            w = query.values()[0]
            return self.logpdf_conditional(w, z)

    # Internal simulators and assesors.

    def simulate_joint(self):
        return self.rng.multivariate_normal(
            [0,0], cov=[[1,1-self.noise],[1-self.noise,1]])

    def logpdf_joint(self, x, y):
        return multivariate_normal.logpdf(
            [x,y], [0,0],
            cov=[[1,1-self.noise],[1-self.noise,1]])

    def logpdf_marginal(self, z):
        return norm.logpdf(z, scale=1)

    def logpdf_conditional(self, w, z):
        mean = self.conditional_mean(z)
        var = self.conditional_variance(z)
        return norm.logpdf(w, loc=mean, scale=np.sqrt(var))

    def simulate_conditional(self, z):
        mean = self.conditional_mean(z)
        var = self.conditional_variance(z)
        return self.rng.normal(loc=mean, scale=np.sqrt(var))

    def conditional_mean(self, z):
        return (1-self.noise)*z

    def conditional_variance(self, z):
        return (1-self.noise**2)

    def mutual_information(self):
        cov = 1-self.noise
        return -.5 * np.log(1-cov**2)
