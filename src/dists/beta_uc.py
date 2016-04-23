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
import scipy

import gpmcc.utils.general as gu
import gpmcc.utils.sampling as su

from gpmcc.dists.distribution import DistributionGpm


class BetaUC(DistributionGpm):
    """Beta distribution with exponential prior on strength and beta
    prior on balance. Uncollapsed.

    s ~ Exponential(mu)
    b ~ Beta(alpha, beta)
    x ~ Beta(s*b, s*(1-b))
    """

    def __init__(self, N=0, sum_log_x=0, sum_minus_log_x=0, strength=None,
            balance=None, mu=1, alpha=.5, beta=.5, distargs=None, rng=None):
        assert mu > 0
        assert alpha > 0
        assert beta > 0
        self.rng = gu.gen_rng() if rng is None else rng
        # Sufficient statistics.
        self.N = N
        self.sum_log_x = sum_log_x
        self.sum_minus_log_x = sum_minus_log_x
        # Hyperparameters.
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        # Parameters.
        self.strength, self.balance = strength, balance
        if strength is None or balance is None:
            self.strength = self.rng.exponential(scale=1./mu)
            self.balance = self.rng.beta(alpha, beta)
            assert self.strength > 0 and 0 < self.balance < 1

    def incorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N += 1.
        self.sum_log_x += log(x)
        self.sum_minus_log_x += log(1.-x)

    def unincorporate(self, x, y=None):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N -= 1.
        if self.N <= 0:
            self.sum_log_x = 0
            self.sum_minus_log_x = 0
        else:
            self.sum_log_x -= log(x)
            self.sum_minus_log_x -= log(1.-x)

    def logpdf(self, x, y=None):
        try: x, y = self.preprocess(x, y, self.get_distargs())
        except ValueError: return -float('inf')
        return BetaUC.calc_predictive_logp(x, self.strength, self.balance)

    def logpdf_marginal(self):
        data_logp = BetaUC.calc_log_likelihood(
            self.N, self.sum_log_x, self.sum_minus_log_x, self.strength,
            self.balance)
        prior_logp = BetaUC.calc_log_prior(
            self.strength, self.balance, self.mu, self.alpha, self.beta)
        return data_logp + prior_logp

    def simulate(self, y=None):
        alpha = self.strength * self.balance
        beta = self.strength * (1. - self.balance)
        return self.rng.beta(alpha, beta)

    def transition_params(self):
        n_samples = 25
        # Transition strength.
        log_pdf_fun_str = lambda strength :\
            BetaUC.calc_log_likelihood(
                self.N, self.sum_log_x, self.sum_minus_log_x, strength,
                self.balance) \
            + BetaUC.calc_log_prior(
                strength, self.balance, self.mu, self.alpha, self.beta)
        self.strength = su.mh_sample(self.strength, log_pdf_fun_str,
            .5, [.0, float('Inf')], burn=n_samples, rng=self.rng)
        # Transition balance.
        log_pdf_fun_bal = lambda balance : \
            BetaUC.calc_log_likelihood(
                self.N, self.sum_log_x, self.sum_minus_log_x, self.strength,
                balance) \
            + BetaUC.calc_log_prior(
                self.strength, balance, self.mu, self.alpha, self.beta)
        self.balance = su.mh_sample(self.balance, log_pdf_fun_bal,
            .25, [0, 1], burn=n_samples, rng=self.rng)

    def set_hypers(self, hypers):
        assert hypers['mu'] > 0
        assert hypers['alpha'] > 0
        assert hypers['beta'] > 0
        self.mu = hypers['mu']
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    def get_hypers(self):
        return {'mu': self.mu, 'alpha': self.alpha, 'beta':
            self.beta}

    def get_params(self):
        return {'balance': self.balance, 'strength': self.strength}

    def get_suffstats(self):
        return {'N': self.N, 'sum_log_x': self.sum_log_x,
            'sum_minus_log_x': self.sum_minus_log_x}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = float(len(X))
        Sx = np.sum(X)
        Mx = np.sum(1-X)
        grids['mu'] = gu.log_linspace(1./N, N, n_grid)
        grids['alpha'] = gu.log_linspace(Sx/N, Sx, n_grid)
        grids['beta'] = gu.log_linspace(Mx/N, Mx, n_grid)
        return grids

    @staticmethod
    def name():
        return 'beta_uc'

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
    def calc_predictive_logp(x, strength, balance):
        assert strength > 0 and balance > 0 and balance < 1
        alpha = strength * balance
        beta = strength * (1.-balance)
        return scipy.stats.beta.logpdf(x, alpha, beta)

    @staticmethod
    def calc_log_likelihood(N, sum_log_x, sum_minus_log_x, strength,
            balance):
        assert strength > 0 and balance > 0 and balance < 1
        alpha = strength * balance
        beta = strength * (1. - balance)
        lp = 0
        lp -= N * scipy.special.betaln(alpha, beta)
        lp += (alpha - 1.) * sum_log_x
        lp += (beta - 1.) * sum_minus_log_x
        assert not np.isnan(lp)
        return lp

    @staticmethod
    def calc_log_prior(strength, balance, mu, alpha, beta):
        assert strength > 0 and balance > 0 and balance < 1
        log_strength = scipy.stats.expon.logpdf(strength, scale=1./mu)
        log_balance = scipy.stats.beta.logpdf(balance, alpha, beta)
        return log_strength + log_balance

    @staticmethod
    def preprocess(x, y, distargs=None):
        if not 0 < x < 1:
            raise ValueError('Beta requires (0,1): %s' % str(x))
        return x, y
