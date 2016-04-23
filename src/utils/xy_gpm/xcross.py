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
from scipy.stats import multivariate_normal
from scipy.stats import norm

from gpmcc.utils.xy_gpm import synthetic


class XCrossGpm(synthetic.SyntheticXyGpm):
    """Y = (+/- w.p .5) X + N(0,noise)."""

    def simulate_xy(self, size=None):
        X = np.zeros((size,2))
        for i in xrange(size):
            if self.rng.rand() < .5:
                cov = np.array([[1,1-self.noise],[1-self.noise,1]])
            else:
                cov = np.array([[1,-1+self.noise],[-1+self.noise,1]])
            X[i,:] = self.rng.multivariate_normal([0,0], cov=cov)
        return X

    def logpdf_xy(self, x, y):
        return logsumexp([
            np.log(.5)+multivariate_normal.logpdf([x,y], [0,0],
                cov=[[1,1-self.noise],[1-self.noise,1]]),
            np.log(.5)+multivariate_normal.logpdf([x,y], [0,0],
                cov=[[1,-1+self.noise],[-1+self.noise,1]]),
            ])

    def logpdf_x(self, x):
        return norm.logpdf(x, scale=1)

    def logpdf_y(self, y):
        return norm.logpdf(y, scale=1)

    def logpdf_x_given_y(self, x, y):
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        raise NotImplementedError
