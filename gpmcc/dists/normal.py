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
import warnings
from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2*math.pi)

class Normal(DistributionGpm):
    """Normal distribution with normal prior on mean and gamma prior on
    precision. Collapsed.

    rho ~ Gamma(nu/2, s/2)
    mu ~ Normal(m, r*rho)
    x ~ Normal(mu, rho)

    http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
    Note that Teh uses Normal-InverseGamma to mean Normal-Gamma for prior.
    """

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, m=0, r=1, s=1, nu=1,
            distargs=None):
        assert s > 0.
        assert r > 0.
        assert nu > 0.
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        self.sum_x_sq = sum_x_sq
        # Hyper parameters.
        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

    def incorporate(self, x):
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        if self.N == 0:
            self.sum_x = 0.0
            self.sum_x_sq = 0.0
        else:
            self.sum_x -= x
            self.sum_x_sq -= x*x

    def logpdf(self, x):
        return Normal.calc_predictive_logp(x, self.N, self.sum_x,
            self.sum_x_sq, self.m, self.r, self.s, self.nu)

    def logpdf_singleton(self, x):
        return Normal.calc_predictive_logp(x, 0, 0, 0, self.m, self.r,
            self.s, self.nu)

    def logpdf_marginal(self):
        return Normal.calc_logpdf_marginal(self.N, self.sum_x, self.sum_x_sq,
            self.m, self.r, self.s, self.nu)

    def simulate(self):
        rn, nun, mn, sn = Normal.posterior_hypers(self.N, self.sum_x,
            self.sum_x_sq, self.m, self.r, self.s, self.nu)
        mu, rho = Normal.sample_parameters(mn, rn, sn, nun)
        return np.random.normal(loc=mu, scale=rho**-.5)

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

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x, 'sum_x_sq': self.sum_x_sq}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        # Plus 1 for single observation case.
        N = len(X) + 1.
        ssqdev = np.var(X) * len(X) + 1.
        # Data dependent heuristics.
        grids['m'] = np.linspace(min(X), max(X) + 5, n_grid)
        grids['r'] = gu.log_linspace(1. / N, N, n_grid)
        grids['s'] = gu.log_linspace(ssqdev / 100., ssqdev, n_grid)
        grids['nu'] = gu.log_linspace(1., N, n_grid) # df >= 1
        return grids

    @staticmethod
    def name():
        return 'normal'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_hypers(N, sum_x, sum_x_sq, m, r,
            s, nu)
        rm, num, mm, sm = Normal.posterior_hypers(N+1, sum_x+x,
            sum_x_sq+x*x, m, r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        ZM = Normal.calc_log_Z(rm, sm, num)
        return -.5 * LOG2PI + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_hypers(N, sum_x, sum_x_sq, m, r,
            s, nu)
        Z0 = Normal.calc_log_Z(r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        return -(N/2.) * LOG2PI + ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + float(N)
        nun = nu + float(N)
        mn = (r*m + sum_x)/rn
        sn = s + sum_x_sq + r*m*m - rn*mn*mn
        if sn == 0:
            warnings.warn('Posterior_update_parameters: sn truncated.')
            sn = s
        return rn, nun, mn, sn

    @staticmethod
    def calc_log_Z(r, s, nu):
        return ((nu + 1.) / 2.) * LOG2 + .5 * LOGPI - .5 * log(r) \
                - (nu / 2.) * log(s) + gammaln(nu/2.0)

    @staticmethod
    def sample_parameters(m, r, s, nu):
        rho = np.random.gamma(nu/2., scale=2./s)
        mu = np.random.normal(loc=m, scale=1./(rho*r)**.5)
        return mu, rho
