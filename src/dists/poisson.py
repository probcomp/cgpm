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

from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Poisson(DistributionGpm):
    """Poisson distribution with gamma prior on mu. Collapsed.

    mu ~ Gamma(a, b)
    x ~ Poisson(mu)
    """

    def __init__(self, N=0, sum_x=0, sum_log_fact_x=0, a=1, b=1,
            distargs=None):
        assert a > 0
        assert b > 0
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        self.sum_log_fact_x = sum_log_fact_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def incorporate(self, x):
        self.N += 1.0
        self.sum_x += x
        self.sum_log_fact_x += gammaln(x+1)

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        self.sum_x -= x
        self.sum_log_fact_x -= gammaln(x+1)

    def logpdf(self, x):
        return Poisson.calc_predictive_logp(x, self.N, self.sum_x,
            self.sum_log_fact_x, self.a, self.b)

    def logpdf_marginal(self):
        return Poisson.calc_logpdf_marginal(self.N, self.sum_x,
            self.sum_log_fact_x, self.a, self.b)

    def logpdf_singleton(self, x):
        return Poisson.calc_predictive_logp(x, 0, 0, 0, self.a,
            self.b)

    def simulate(self):
        an, bn = Poisson.posterior_hypers(self.N, self.sum_x,
            self.a, self.b)
        draw = np.random.negative_binomial(an, bn/(bn+1.))
        return draw

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.a = hypers['a']
        self.b = hypers['b']

    def get_hypers(self):
        return {
            'a': self.a,
            'b': self.b
        }

    def get_suffstats(self):
        return {
            'N': self.N,
            'sum_x' : self.sum_x,
            'sum_log_fact_x': self.sum_log_fact_x
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        # only use integers for a so we can nicely draw from a negative binomial
        # in predictive_draw
        grids['a'] = np.unique(np.round(np.linspace(1, len(X), n_grid)))
        grids['b'] = gu.log_linspace(.1, float(len(X)), n_grid)
        return grids

    @staticmethod
    def name():
        return 'poisson'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, sum_log_fact_x, a, b):
        if float(x) != x or x < 0:
            return float('-inf')
        an, bn = Poisson.posterior_hypers(N, sum_x, a, b)
        am, bm = Poisson.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Poisson.calc_log_Z(an, bn)
        ZM = Poisson.calc_log_Z(am, bm)
        return  ZM - ZN - gammaln(x+1)

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, sum_log_fact_x, a, b):
        an, bn = Poisson.posterior_hypers(N, sum_x, a, b)
        Z0 = Poisson.calc_log_Z(a, b)
        ZN = Poisson.calc_log_Z(an, bn)
        return ZN - Z0 - sum_log_fact_x

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + sum_x
        bn = b + N
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  gammaln(a)-a*log(b)
        return Z
