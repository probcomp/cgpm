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

from gpmcc.utils.xy_gpm import synthetic

class DotsGpm(synthetic.SyntheticXyGpm):
    """(X,Y) ~ Four Dots"""

    def simulate_xy(self, size=None):
        X = np.zeros((size,2))
        mx = [ -1, 1, -1, 1]
        my = [ -1, -1, 1, 1]
        for i in xrange(size):
            n = self.rng.randint(4)
            x = self.rng.normal(loc=mx[n], scale=self.noise)
            y = self.rng.normal(loc=my[n], scale=self.noise)
            X[i,0] = x
            X[i,1] = y
        return X

    def logpdf_xy(self, x, y):
        raise NotImplementedError

    def logpdf_x(self, x):
        raise NotImplementedError

    def logpdf_y(self, y):
        raise NotImplementedError

    def logpdf_x_given_y(self, x, y):
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        raise NotImplementedError
