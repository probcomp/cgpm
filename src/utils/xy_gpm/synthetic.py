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

from gpmcc.utils.general import gen_rng

class SyntheticXyGpm(object):
    """Interface synthetic, two-dimensional GPMs that take arbitrary noise
    parameter in (0,1).

    Typically used for simulating and evaluating the log density of a zero
    correlation dataset, but any distribution over R2 is possible.
    """

    def __init__(self, noise=.1, rng=None):
        """Initialize the Gpm with given noise parameter.

        Parameters
        ----------
        noise : float
            Value in (0,1) indicating the noise level of the distribution.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        assert 0 < noise < 1
        if rng is None:
            rng = gen_rng(0)
        self.rng = rng
        self.noise = noise

    def simulate_xy(self, size=None):
        """Simulate from the joint distribution (X,Y)."""
        raise NotImplementedError

    def logpdf_xy(self, x, y):
        """Evaluate the joint log density p(x,y)."""
        raise NotImplementedError

    def logpdf_x(self, x):
        """Evaluate the marginal log density p(x,y)."""
        raise NotImplementedError

    def logpdf_y(self, y):
        """Evaluate the marginal log density p(y)."""
        raise NotImplementedError

    def simulate_x_given_y(self, y):
        """Simulate from the conditional density p(x|y)."""
        raise NotImplementedError

    def logpdf_x_given_y(self, x, y):
        """Evaluate the conditional log density p(x|y)."""
        raise NotImplementedError

    def simulate_y_given_x(self, x):
        """Simulate from the conditional density p(y|x)."""
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        """Evaluate the marginal log density p(y|x)."""
        raise NotImplementedError

    def mutual_information(self):
        """Compute the mutual information MI(X:Y)"""
        raise NotImplementedError
