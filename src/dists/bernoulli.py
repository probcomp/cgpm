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

from scipy.special import betaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Bernoulli(DistributionGpm):
    """Bernoulli distribution with beta prior on bias theta.

    theta ~ Beta(alpha, beta)
    x ~ Bernoulli(theta)
    """

    def __init__(self, N=0, x_sum=0, alpha=1, beta=1, distargs=None, rng=None):
        assert alpha > 0
        assert beta > 0
        self.rng = gu.gen_rng() if rng is None else rng
        self.distargs = {'k':2}
        # Sufficent statistics.
        self.N = N
        self.x_sum = x_sum
        # Hyperparameters.
        self.alpha = alpha
        self.beta = beta

    def incorporate(self, x, y=None):
        assert x == 1.0 or x == 0.0
        self.N += 1
        self.x_sum += x

    def unincorporate(self, x, y=None):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        assert x == 1. or x == 0.
        self.N -= 1
        self.x_sum -= x

    def logpdf(self, x, y=None):
        return Bernoulli.calc_predictive_logp(
            x, self.N, self.x_sum, self.alpha, self.beta)

    def logpdf_marginal(self):
        return Bernoulli.calc_logpdf_marginal(
            self.N, self.x_sum, self.alpha, self.beta)

    def simulate(self, y=None):
        p0 = Bernoulli.calc_predictive_logp(
            0, self.N, self.x_sum, self.alpha, self.beta)
        p1 = Bernoulli.calc_predictive_logp(
            1, self.N, self.x_sum, self.alpha, self.beta)
        return gu.log_pflip([p0, p1], rng=self.rng)

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
        return {'N':self.N, 'x_sum':self.x_sum}

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
    def calc_predictive_logp(x, N, x_sum, alpha, beta):
        if int(x) not in [0, 1]:
            return float('-inf')
        log_denom = log(N + alpha + beta)
        if x == 1.0:
            return log(x_sum + alpha) - log_denom
        else:
            return log(N - x_sum + beta) - log_denom

    @staticmethod
    def calc_logpdf_marginal(N, x_sum, alpha, beta):
        return betaln(x_sum + alpha, N - x_sum + beta) - betaln(alpha, beta)
