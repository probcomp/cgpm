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
from scipy.integrate import quad

from gpmcc.utils.xy_gpm import synthetic

class SinGpm(synthetic.SyntheticXyGpm):
    """Y = cos(X) + Noise."""

    # Domain of cos(X).
    D = [-3.*np.pi/2., 3.*np.pi/2.]

    def simulate_xy(self, size=None):
        X = np.zeros((size,2))
        for i in xrange(size):
            x = self.rng.uniform(self.D[0], self.D[1])
            if np.cos(x) < 0:
                error = self.rng.uniform(0, self.noise)
            else:
                error = -self.rng.uniform(0, self.noise)
            X[i,0] = x
            X[i,1] = np.cos(x) + error
        return X

    def logpdf_xy(self, x, y):
        if not self.D[0] <= x <= self.D[1]:
            return -float('inf')
        if np.cos(x) < 0 and not np.cos(x) <= y <= np.cos(x) + self.noise:
            return -float('inf')
        if np.cos(x) > 0 and not np.cos(x) - self.noise <= y <= np.cos(x):
            return -float('inf')
        return -np.log(self.D[1]-self.D[0]) - np.log(self.noise)

    def logpdf_x(self, x):
        if not self.D[0] <= x <= self.D[1]:
            return -float('inf')
        return -np.log(self.D[1]-self.D[0])

    def logpdf_y(self, y):
        # XXX Figure out why this does not integrate to one over y \in [-1, 1]
        def integrand(x):
            return (np.cos(x)<0 and np.cos(x) <= y <= np.cos(x) + self.noise) \
                or (np.cos(x)>0 and np.cos(x) - self.noise <= y <= np.cos(x))
        length = quad(integrand, self.D[0], self.D[1])[0]
        return -np.log(self.noise) - np.log(self.D[1]-self.D[0]) + np.log(length)

    def logpdf_x_given_y(self, x, y):
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        raise NotImplementedError
