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
from scipy.stats import norm

from gpmcc.utils.xy_gpm import synthetic


class DotsGpm(synthetic.SyntheticXyGpm):
    """(X,Y) ~ Four Dots"""

    mx = [ -1, 1, -1, 1]
    my = [ -1, -1, 1, 1]

    def simulate_xy(self, size=None):
        X = np.zeros((size,2))
        for i in xrange(size):
            n = self.rng.randint(4)
            x = self.rng.normal(loc=self.mx[n], scale=self.noise)
            y = self.rng.normal(loc=self.my[n], scale=self.noise)
            X[i,0] = x
            X[i,1] = y
        return X

    def logpdf_xy(self, x, y):
        return logsumexp([np.log(.25)
                + norm.logpdf(x, loc=mx, scale=self.noise)
                + norm.logpdf(y, loc=my, scale=self.noise)
            for (mx,my) in zip(self.mx, self.my)])

    def logpdf_x(self, x):
        return logsumexp([np.log(.5) + norm.logpdf(x, loc=mx, scale=self.noise)
            for mx in set(self.mx)])

    def logpdf_y(self, y):
        return logsumexp([np.log(.5) + norm.logpdf(y, loc=my, scale=self.noise)
            for my in set(self.my)])

    def logpdf_x_given_y(self, x, y):
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        raise NotImplementedError

    def mutual_information(self):
        samples = self.simulate_xy(size=1000)
        return np.mean([self.logpdf_xy(x,y) - self.logpdf_x(x) -
            self.logpdf_y(y) for [x,y] in samples])
