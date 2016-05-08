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

import numpy as np

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import uniform

from gpmcc.dists.distribution import DistributionGpm
from gpmcc.utils import general as gu
from gpmcc.utils import sampling as su


class NormalTrunc(DistributionGpm):
    """Normal distribution with normal prior on mean and gamma prior on
    precision. Uncollapsed.
    sigma ~ Gamma(shape=alpha, scale=beta)
    mu ~ Uniform(low, high)
    X ~ Normal(mu, sigma)
    """

    def __init__(self, hypers=None, params=None, distargs=None, rng=None):
        self.rng = gu.gen_rng() if rng is None else rng
        # Distargs
        self.l = distargs['l']
        self.h = distargs['h']
        # Sufficient statistics.
        self.N = 0
        self.sum_x = 0
        self.sum_x_sq = 0
        # Hyperparameters (fixed).
        self.alpha = 2.
        self.beta = 2.
        # Uncollapsed mean and precision parameters.
        if params is None: params = {}
        self.mu = params.get('mu', None)
        self.sigma = params.get('sigma', 1)
        if not self.mu or not self.sigma:
            self.mu, self.sigma = NormalTrunc.sample_parameters(
                self.alpha, self.beta, self.l, self.h, self.rng)

    def incorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def unincorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
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
        try: x, y = self.preprocess(x, y, self.get_distargs())
        except ValueError: return -float('inf')
        logpdf_unorm = NormalTrunc.calc_predictive_logp(
            x, self.mu, self.sigma, self.l, self.h)
        logcdf_norm = NormalTrunc.calc_log_normalizer(
            self.mu, self.sigma, self.l, self.h)
        return logpdf_unorm - logcdf_norm

    def logpdf_score(self):
        data_logp = NormalTrunc.calc_log_likelihood(
            self.N, self.sum_x, self.sum_x_sq, self.sigma, self.mu)
        prior_logp = NormalTrunc.calc_log_prior(
            self.mu, self.sigma, self.alpha, self.beta, self.l, self.h)
        normalizer_logp = NormalTrunc.calc_log_normalizer(
            self.mu, self.sigma, self.l, self.h)
        return data_logp + prior_logp - self.N * normalizer_logp

    def simulate(self, y=None):
        max_iters = 1000
        for _ in xrange(max_iters):
            x = self.rng.normal(loc=self.mu, scale=self.sigma)
            if self.l <= x <= self.h:
                return x
        else:
            raise RuntimeError('NormalTrunc failed to rejection sample.')

    def transition_params(self):
        n_samples = 100

        # Transition mu.
        fn_logpdf_mu = lambda mu : NormalTrunc.calc_log_likelihood(
            self.N, self.sum_x, self.sum_x_sq, self.sigma, mu)

        self.mu = su.mh_sample(
            self.mu, fn_logpdf_mu, 1, [self.l, self.h],
            burn=n_samples, rng=self.rng)

        # Transition sigma.
        fn_logpdf_sigma = lambda sigma : NormalTrunc.calc_log_likelihood(
            self.N, self.sum_x, self.sum_x_sq, sigma, self.mu) + \
            NormalTrunc.calc_log_normalizer(self.mu, sigma, self.l, self.h)

        self.sigma = su.mh_sample(
            self.sigma, fn_logpdf_sigma, 1, [0, float('inf')], burn=n_samples,
            rng=self.rng)

    def set_hypers(self, hypers):
        return

    def get_hypers(self):
        return {}

    def get_params(self):
        return {'mu': self.mu, 'sigma': self.sigma}

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x, 'sum_x_sq': self.sum_x_sq}

    def get_distargs(self):
        return {'l':self.l, 'h':self.h}

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
        return np.log(
            norm.cdf(h, loc=mu, scale=sigma)
            - norm.cdf(l, loc=mu, scale=sigma))

    @staticmethod
    def calc_predictive_logp(x, mu, sigma, l, h):
        return norm.logpdf(x, loc=mu, scale=sigma)

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, sigma, mu):
        return (
            - N/2. * np.log(2*np.pi)
            - N * np.log(sigma)
            - 1 / (2 * sigma * sigma)
            * (N*mu**2 - 2*mu*sum_x + sum_x_sq))

    @staticmethod
    def calc_log_prior(mu, sigma, alpha, beta, l, h):
        log_sigma = gamma.logpdf(sigma, alpha, scale=beta)
        log_mu = uniform.logpdf(mu, loc=l, scale=l+h)
        return log_mu + log_sigma

    @staticmethod
    def sample_parameters(alpha, beta, l, h, rng):
        sigma = rng.gamma(alpha, scale=beta)
        mu = rng.uniform(low=l, high=h)
        return mu, sigma

    @staticmethod
    def preprocess(x, y, distargs=None):
        l, h = distargs['l'], distargs['h']
        if not l <= x <= h:
            raise ValueError('NormalTrunc requires [{},{}]: {}'.format(l, h, x))
        return x, y

