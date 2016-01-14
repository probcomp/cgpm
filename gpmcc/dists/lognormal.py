# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
from math import log

import numpy as np

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm
from gpmcc.dists.normal import Normal

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2.0*math.pi)

class Lognormal(DistributionGpm):
    """Log-normal (zero-bounded) distribution with normal prior on mean and
    gamma prior on precision. Collapsed.

    rho ~ Gamma(a, b)
    mu ~ Normal(m, t)
    x ~ Lognormal(mu, rho)
    """

    def __init__(self, N=0, sum_log_x=0, sum_log_x_sq=0, m=1, r=1, s=1,
            nu=0, distargs=None):
        assert r > 0.
        assert s > 0.
        assert nu > 0.
        # Sufficient statistics.
        self.N = N
        self.sum_log_x_sq = sum_log_x_sq
        self.sum_log_x = sum_log_x
        # Hyperparameters.
        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

    def incorporate(self, x):
        if x <= 0:
            raise ValueError('Lognormal requires positive observations.')
        self.N += 1.0
        self.sum_log_x += log(x)
        self.sum_log_x_sq += log(x) * log(x)

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        if x <= 0:
            raise ValueError('Lognormal requires positive observations.')
        self.N -= 1.0
        self.sum_log_x -= log(x)
        self.sum_log_x_sq -= log(x) * log(x)

    def predictive_logp(self, x):
        if x < 0:
            return float('-inf')
        return -log(x) + \
            Normal.calc_predictive_logp(log(x), self.N, self.sum_log_x,
                self.sum_log_x_sq, self.m, self.r, self.s, self.nu)

    def marginal_logp(self):
        return -self.sum_log_x + \
            Normal.calc_marginal_logp(self.N, self.sum_log_x,
                self.sum_log_x_sq, self.m, self.r, self.s, self.nu)

    def singleton_logp(self, x):
        if x < 0:
            return float('-inf')
        return - log(x) + \
            Normal.calc_predictive_logp(log(x), 0, 0, 0, self.m, self.r,
                self.s, self.nu)

    def simulate(self):
        # XXX This implementation is not verified but will be covered in
        # future univariate simulate tests, see Github issue #14.
        # Simulate normal parameters
        rn, nun, mn, sn = Normal.posterior_hypers(self.N, self.sum_log_x,
            self.sum_log_x_sq, self.m, self.r, self.s, self.nu)
        mu, rho = Normal.sample_parameters(mn, rn, sn, nun)
        x = np.random.norm(loc=mu, scale=rho**-.5)
        return np.exp(x)

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['r'] > 0.
        assert hypers['s'] > 0.
        assert hypers['nu'] > 0.
        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def get_hypers(self):
        return { 'm': self.m, 'r': self.r, 's': self.s, 'nu': self.nu }

    def get_suffstats(self):
        return {'N': self.N, 'sum_log_x': self.sum_log_x,
            'sum_log_x_sq': self.sum_log_x_sq}

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        ssqdev = np.var(X) * float(len(X));
        grids['m'] = gu.log_linspace(1e-4, max(X), n_grid)
        grids['r'] = gu.log_linspace(.1, float(len(X)), n_grid)
        grids['nu'] = gu.log_linspace(.1, float(len(X)), n_grid)
        grids['s'] = gu.log_linspace(.1, float(len(X)), n_grid)
        return grids

    @staticmethod
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
        return Normal.plot_dist(X, clusters, ax=ax, Y=Y, hist=hist)

    @staticmethod
    def name():
        return 'lognormal'

    @staticmethod
    def is_collapsed():
        return True
