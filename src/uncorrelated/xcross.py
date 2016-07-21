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

from cgpm.uncorrelated.undirected import UnDirectedXyGpm
from cgpm.utils import general as gu
from cgpm.utils import mvnormal as multivariate_normal


class XCross(UnDirectedXyGpm):
    """Y = (+/- w.p .5) X + N(0,noise)."""

    def simulate_joint(self):
        if self.rng.rand() < .5:
            cov = np.array([[1,1-self.noise],[1-self.noise,1]])
        else:
            cov = np.array([[1,-1+self.noise],[-1+self.noise,1]])
        return self.rng.multivariate_normal([0,0], cov=cov)

    def logpdf_joint(self, x, y):
        X = np.array([x, y])
        Mu = np.array([0, 0])
        Sigma0 = np.array([[1, 1 - self.noise], [1 - self.noise, 1]])
        Sigma1 = np.array([[1, -1 + self.noise], [-1 + self.noise, 1]])
        return gu.logsumexp([
            np.log(.5)+multivariate_normal.logpdf(X, Mu, Sigma0),
            np.log(.5)+multivariate_normal.logpdf(X, Mu, Sigma1),
        ])
