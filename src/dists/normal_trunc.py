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
from scipy.stats import norm, gamma

from gpmcc.dists.distribution import DistributionGpm
from gpmcc.dists.normal import Normal
from gpmcc.utils import general as gu
from gpmcc.utils import sampling as su

LOG2 = log(2.0)
LOGPI = log(np.pi)
LOG2PI = log(2*np.pi)

class NormalTrunc(DistributionGpm):
    """Normal distribution with normal prior on mean and gamma prior on
    precision. Uncollapsed.
    sigma ~ Gamma(shape=1, scale=.5)
    mu ~ Uniform(low, high)
    X ~ Normal(mu, sigma)
    """

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, mu=None, sigma=None,
            distargs=None):
        # Distargs
        self.l = distargs['l']
        self.h = distargs['h']
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        self.sum_x_sq = sum_x_sq
        # Uncollapsed mean and precision parameters.
        self.mu, self.sigma = mu, sigma
        if mu is None or sigma is None:
            self.mu, self.sigma = NormalTrunc.sample_parameters(self.l, self.h)

    def incorporate(self, x, y=None):
        assert self.l<=x<=self.h
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def unincorporate(self, x, y=None):
        assert self.l<=x<=self.h
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        if self.N == 0:
            self.sum_x = 0.0
            self.sum_x_sq = 0.0
        else:
            self.sum_x -= x
            self.sum_x_sq -= x*x

    def logpdf(self, x, y=None):
        if not self.l<=x<=self.h:
            return float('-inf')
        logpdf_unorm = NormalTrunc.calc_predictive_logp(x, self.mu, self.sigma)
        logcdf_norm = NormalTrunc.calc_log_normalizer(
            self.mu, self.sigma, self.l, self.h)
        return logpdf_unorm - logcdf_norm

    def logpdf_marginal(self):
        data_logp = NormalTrunc.calc_log_likelihood(self.N, self.sum_x,
            self.sum_x_sq, self.sigma, self.mu)
        prior_logp = NormalTrunc.calc_log_prior(self.mu, self.sigma,
            self.l, self.sigma)
        normalizer_logp = NormalTrunc.calc_log_normalizer(self.mu, self.sigma,
            self.l, self.h)
        return data_logp + prior_logp - self.N * normalizer_logp

    def simulate(self, y=None):
        max_iters = 1000
        for _ in xrange(max_iters):
            x = norm.rvs(loc=self.mu, scale=self.sigma)
            if self.l <= x <= self.h:
                return x
        else:
            raise RuntimeError('NormalTrunc failed to rejection sample.')

    def transition_params(self):
        n_samples = 30

        # Transition mu.
        fn_logpdf_mu = lambda mu :\
          NormalTrunc.calc_log_likelihood(
                self.N, self.sum_x, self.sum_x_sq, self.sigma, mu) \

        self.mu = su.mh_sample(
            self.mu, fn_logpdf_mu, 1, [self.l, self.h],
            burn=n_samples)

        # Transition sigma.
        fn_logpdf_sigma = lambda sigma :\
          NormalTrunc.calc_log_likelihood(self.N, self.sum_x, self.sum_x_sq,
            sigma, self.mu) \

        self.sigma = su.mh_sample(
            self.sigma, fn_logpdf_sigma, 1, [0, float('inf')], burn=n_samples)

    def set_hypers(self, hypers):
        return

    def get_hypers(self):
        return {}

    def get_params(self):
        return {'mu': self.mu, 'sigma': self.sigma}

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x, 'sum_x_sq': self.sum_x_sq}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        return dict()

    @staticmethod
    def name():
        return 'normal_trunc'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_log_normalizer(mu, sigma, l, h):
        return np.log(norm.cdf(h, loc=mu, scale=sigma)
            - norm.cdf(l, loc=mu, scale=sigma))

    @staticmethod
    def calc_predictive_logp(x, mu, sigma):
        return norm.logpdf(x, loc=mu, scale=sigma)

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, sigma, mu):
        return -(N/2.)*LOG2PI - N*log(sigma) \
            - 1/(2*sigma*sigma) * (N*mu**2 - 2*mu*sum_x + sum_x_sq)

    @staticmethod
    def calc_log_prior(mu, sigma, l, h):
        log_sigma = gamma.logpdf(sigma, 1, scale=.5)
        log_mu = -np.log(h-l)
        return log_mu + log_sigma

    @staticmethod
    def sample_parameters(l, h):
        sigma = np.random.gamma(1, scale=.5)
        mu = np.random.uniform(low=l, high=h)
        return mu, sigma
