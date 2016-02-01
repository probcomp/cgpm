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

class Categorical(DistributionGpm):
    """Categorical distribution with symmetric dirichlet prior on
    category weight vector v.

    k := distarg
    v ~ Symmetric-Dirichlet(alpha/k)
    x ~ Categorical(v)
    http://www.cs.berkeley.edu/~stephentu/writeups/dirichlet-conjugate-prior.pdf
    """

    def __init__(self, N=0, counts=None, alpha=1, distargs=None):
        # Number of categories.
        assert float(distargs['k']) == int(distargs['k'])
        self.k = int(distargs['k'])
        # Sufficient statistics.
        self.N = N
        if counts is None:
            self.counts = np.zeros(self.k)
        else:
            assert self.k == len(counts)
            self.counts = np.asarray(counts)
        # Hyperparameter.
        self.alpha = alpha

    def incorporate(self, x):
        if not Categorical.validate(x, self.k):
            raise ValueError('Invalid categorical observation %s.' % str(x))
        self.N += 1
        self.counts[int(x)] += 1

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        if not Categorical.validate(x, self.k):
            raise ValueError('Invalid categorical observation removed.')
        self.N -= 1
        self.counts[int(x)] -= 1

    def logpdf(self, x):
        return Categorical.calc_predictive_logp(x, self.N, self.counts,
            self.alpha)

    def logpdf_marginal(self):
        return Categorical.calc_logpdf_marginal(self.N, self.counts,
            self.alpha)

    def singleton_logp(self, x):
        return Categorical.calc_predictive_logp(x, 0, [0]*self.k,
            self.alpha)

    def simulate(self):
        return gu.pflip(self.counts + self.alpha)

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def get_hypers(self):
        return {
            'alpha': self.alpha,
        }

    def get_suffstats(self):
        return {
            'N' : self.N,
            'counts' : self.counts
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def name():
        return 'categorical'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return False

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def validate(x, K):
        assert int(x) == float(x)
        assert 0 <= x and x < K
        return int(x) == float(x) and 0 <= x and x < K

    @staticmethod
    def calc_predictive_logp(x, N, counts, alpha):
        if not Categorical.validate(x, len(counts)):
            return float('-inf')
        x = int(x)
        numer = log(alpha + counts[x])
        denom = log(np.sum(counts) + alpha * len(counts))
        return numer - denom

    @staticmethod
    def calc_logpdf_marginal(N, counts, alpha):
        K = len(counts)
        A = K * alpha
        lg = sum(gammaln(counts[k] + alpha) for k in xrange(K))
        return gammaln(A) - gammaln(A+N) + lg - K * gammaln(alpha)
