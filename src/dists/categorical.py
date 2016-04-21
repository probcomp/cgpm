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

from math import log

import numpy as np
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

    def __init__(self, N=0, counts=None, alpha=1, distargs=None, rng=None):
        self.rng = gu.gen_rng() if rng is None else rng
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

    def incorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N += 1
        self.counts[x] += 1

    def unincorporate(self, x, y=None):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N -= 1
        self.counts[x] -= 1

    def logpdf(self, x, y=None):
        try: x, y = self.preprocess(x, y, self.get_distargs())
        except ValueError: return -float('inf')
        return Categorical.calc_predictive_logp(
            x, self.N, self.counts, self.alpha)

    def logpdf_marginal(self):
        return Categorical.calc_logpdf_marginal(
            self.N, self.counts, self.alpha)

    def simulate(self, y=None):
        return gu.pflip(self.counts + self.alpha, rng=self.rng)

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def get_hypers(self):
        return {'alpha': self.alpha}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N' : self.N, 'counts' : list(self.counts)}

    def get_distargs(self):
        return {'k': self.k}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1./float(len(X)), float(len(X)), n_grid)
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
    def validate(x, K):
        return int(x) == float(x) and 0 <= x < K

    @staticmethod
    def calc_predictive_logp(x, N, counts, alpha):
        numer = log(alpha + counts[x])
        denom = log(np.sum(counts) + alpha * len(counts))
        return numer - denom

    @staticmethod
    def calc_logpdf_marginal(N, counts, alpha):
        K = len(counts)
        A = K * alpha
        lg = sum(gammaln(counts[k] + alpha) for k in xrange(K))
        return gammaln(A) - gammaln(A+N) + lg - K * gammaln(alpha)

    @staticmethod
    def preprocess(x, y, distargs=None):
        k = distargs['k']
        if (int(x) != float(x)) or not (0 <= x < k):
            raise ValueError('Categorical requires [0..{}): {}'.format(k, x))
        return int(x), y
