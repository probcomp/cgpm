# -*- coding: utf-8 -*-

# The MIT License (MIT)

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

from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import expon, gamma

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Exponential(DistributionGpm):
    """Exponential distribution with gamma prior on mu. Collapsed.

    mu ~ Gamma(a, b)
    x ~ Exponential(mu)
    """

    def __init__(self, N=0, sum_x=0, a=1, b=1, distargs=None):
        assert a > 0
        assert b > 0
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def incorporate(self, x):
        self.N += 1.0
        self.sum_x += x

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        self.sum_x -= x

    def logpdf(self, x):
        return Exponential.calc_predictive_logp(x, self.N, self.sum_x,
            self.a, self.b)

    def logpdf_marginal(self):
        return Exponential.calc_logpdf_marginal(self.N, self.sum_x, self.a,
            self.b)

    def logpdf_singleton(self, x):
        return Exponential.calc_predictive_logp(x, 0, 0, self.a,
            self.b)

    def simulate(self):
        an, bn = Exponential.posterior_hypers(self.N, self.sum_x, self.a, self.b)
        mu = gamma.rvs(an, scale=1./bn)
        return expon.rvs(scale=1./mu)

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.b = hypers['b']
        self.a = hypers['a']

    def get_hypers(self):
        return {
            'a': self.a,
            'b': self.b,
        }

    def get_suffstats(self):
        return {
            'N': self.N,
            'sum_x': self.sum_x,
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(.5, float(len(X)),
            n_grid)
        grids['b'] = gu.log_linspace(.5, float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def name():
        return 'exponential'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, a, b):
        if x < 0:
            return float('-inf')
        an,bn = Exponential.posterior_hypers(N, sum_x, a, b)
        am,bm = Exponential.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Exponential.calc_log_Z(an, bn)
        ZM = Exponential.calc_log_Z(am, bm)
        return  ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, a, b):
        an, bn = Exponential.posterior_hypers(N, sum_x, a, b)
        Z0 = Exponential.calc_log_Z(a, b)
        ZN = Exponential.calc_log_Z(an, bn)
        return ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + N
        bn = b + sum_x
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  gammaln(a) - a*log(b)
        return Z
