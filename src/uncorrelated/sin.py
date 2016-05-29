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

from scipy.integrate import dblquad
from scipy.integrate import quad

from gpmcc.uncorrelated import synthetic


class SinGpm(synthetic.SyntheticXyGpm):
    """Y = cos(X) + Noise."""

    # XXX Domain of cos(X), do not change!
    D = [-1.5*np.pi, 1.5*np.pi]

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
        if 0 <= y:
            length, overflow = self._valid_x(y)
            length += 2*overflow
        else:
            length, overflow = self._valid_x(-y)
            length = 2*length + overflow
        length *= 2
        return np.log(length) - np.log(self.noise) - np.log(self.D[1]-self.D[0])

    def logpdf_x_given_y(self, x, y):
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        raise NotImplementedError

    def mutual_information(self):
        def mi_integrand(x, y):
            return np.exp(self.logpdf_xy(x,y)) * \
                (self.logpdf_xy(x,y) - self.logpdf_x(x) - self.logpdf_y(y))
        return dblquad(
            lambda y, x: mi_integrand(x,y), self.D[0], self.D[1],
            self._lower_y, self._upper_y)

    def _valid_x(self, y):
        """Compute valid regions of x for y \in [0, 1], with overflow."""
        assert 0<=y<=1
        x_max = np.arccos(y)
        if y+self.noise < 1:
            x_min = np.arccos(y+self.noise)
        else:
            x_min = 0
        # compute overflow
        overflow = 0
        if y < self.noise:
            overflow = np.arccos(y-self.noise) - np.pi / 2
        return x_max - x_min, overflow

    def _lower_y(self, x):
        if np.cos(x) < 0:
            return np.cos(x)
        else:
            return np.cos(x) - self.noise

    def _upper_y(self, x):
        if np.cos(x) < 0:
            return np.cos(x) + self.noise
        else:
            return np.cos(x)

    def _sanity_test(self):
        # Marginal of x integrates to one.
        print quad(lambda x: np.exp(self.logpdf_x(x)), self.D[0], self.D[1])

        # Marginal of y integrates to one.
        print quad(lambda y: np.exp(self.logpdf_y(y)), -1 ,1)

        # Joint of x,y integrates to one; quadrature will fail for small noise.
        print dblquad(
            lambda y,x: np.exp(self.logpdf_xy(x,y)), self.D[0], self.D[1],
            lambda x: -1, lambda x: 1)
