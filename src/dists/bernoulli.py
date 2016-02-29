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

from scipy.special import betaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Bernoulli(DistributionGpm):
    """Bernoulli distribution with beta prior on bias theta.

    theta ~ Beta(alpha, beta)
    x ~ Bernoulli(theta)
    """

    def __init__(self, N=0, k=0, alpha=1, beta=1, distargs=None):
        assert alpha > 0
        assert beta > 0
        # Sufficient statistics.
        self.N = N
        self.k = k
        # Hyperparameter.
        self.alpha = alpha
        self.beta = beta

    def incorporate(self, x, y=None):
        assert x == 1.0 or x == 0.0
        self.N += 1
        self.k += x

    def unincorporate(self, x, y=None):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        assert x == 1. or x == 0.
        self.N -= 1
        self.k -= x

    def logpdf(self, x, y=None):
        return Bernoulli.calc_predictive_logp(x, self.N, self.k, self.alpha,
            self.beta)

    def logpdf_marginal(self):
        return Bernoulli.calc_logpdf_marginal(self.N, self.k, self.alpha,
            self.beta)

    def simulate(self, y=None):
        p0 = Bernoulli.calc_predictive_logp(0, self.N, self.k, self.alpha,
            self.beta)
        p1 = Bernoulli.calc_predictive_logp(1, self.N, self.k, self.alpha,
            self.beta)
        return gu.log_pflip([p0, p1])

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        assert hypers['beta'] > 0
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    def get_hypers(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N' : self.N, 'k' : self.k}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        grids['beta'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def name():
        return 'bernoulli'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return False

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, k, alpha, beta):
        if int(x) not in [0, 1]:
            return float('-inf')
        log_denom = log(N + alpha + beta)
        if x == 1.0:
            return log(k + alpha) - log_denom
        else:
            return log(N - k + beta) - log_denom

    @staticmethod
    def calc_logpdf_marginal(N, k, alpha, beta):
        return betaln(k + alpha, N - k + beta) - betaln(alpha, beta)
