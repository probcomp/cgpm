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

class RingGpm(synthetic.SyntheticXyGpm):
    """(X,Y) ~ Ring + Noise."""

    def simulate_xy(self, size=None):
        X = np.zeros((size,2))
        for i in xrange(size):
            angle = self.rng.uniform(0., 2.*np.pi)
            distance = self.rng.uniform(1.-self.noise, 1.)
            X[i,0] = np.cos(angle)*distance
            X[i,1] = np.sin(angle)*distance
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
