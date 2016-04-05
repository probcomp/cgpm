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
            nu=1, distargs=None, rng=None):
        assert r > 0.
        assert s > 0.
        assert nu > 0.
        self.rng = gu.gen_rng() if rng is None else rng
        # Sufficient statistics.
        self.N = N
        self.sum_log_x_sq = sum_log_x_sq
        self.sum_log_x = sum_log_x
        # Hyperparameters.
        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

    def incorporate(self, x, y=None):
        if x <= 0:
            raise ValueError('Lognormal requires positive observations.')
        self.N += 1.0
        self.sum_log_x += log(x)
        self.sum_log_x_sq += log(x) * log(x)

    def unincorporate(self, x, y=None):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        if x <= 0:
            raise ValueError('Lognormal requires positive observations.')
        self.N -= 1.0
        self.sum_log_x -= log(x)
        self.sum_log_x_sq -= log(x) * log(x)

    def logpdf(self, x, y=None):
        if x < 0:
            return float('-inf')
        return -log(x) + \
            Normal.calc_predictive_logp(log(x), self.N, self.sum_log_x,
                self.sum_log_x_sq, self.m, self.r, self.s, self.nu)

    def logpdf_marginal(self):
        return -self.sum_log_x + \
            Normal.calc_logpdf_marginal(self.N, self.sum_log_x,
                self.sum_log_x_sq, self.m, self.r, self.s, self.nu)

    def simulate(self, y=None):
        # XXX This implementation is not verified but will be covered in
        # future univariate simulate tests, see Github issue #14.
        # Simulate normal parameters
        mn, rn, sn, nun = Normal.posterior_hypers(self.N, self.sum_log_x,
            self.sum_log_x_sq, self.m, self.r, self.s, self.nu)
        mu, rho = Normal.sample_parameters(mn, rn, sn, nun)
        x = self.rng.normal(loc=mu, scale=rho**-.5)
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
        return {'m': self.m, 'r': self.r, 's': self.s, 'nu': self.nu}

    def get_params(self):
        return {}

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
    def name():
        return 'lognormal'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True
