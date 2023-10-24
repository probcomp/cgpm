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

from builtins import str
from math import log

import numpy as np
import scipy

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu
from cgpm.utils import sampling as su


class Beta(DistributionGpm):
    """Beta distribution with exponential prior on strength and beta
    prior on balance. Uncollapsed.

    s ~ Exponential(mu)
    b ~ Beta(alpha, beta)
    x ~ Beta(s*b, s*(1-b))
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Sufficient statistics.
        self.N = 0
        self.sum_log_x = 0
        self.sum_minus_log_x = 0
        # Hyperparameters (fixed).
        self.mu = 5.
        self.alpha = 1.
        self.beta = 1.
        # Parameters.
        if params is None: params = {}
        self.strength = params.get('strength', None)
        self.balance = params.get('balance', 1)
        if not self.strength or not self.balance:
            self.strength, self.balance = Beta.sample_parameters(
                self.mu, self.alpha, self.beta, self.rng)
        assert self.mu > 0
        assert self.alpha > 0
        assert self.beta > 0

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if np.allclose(0, x):
            x = 0.001
        elif np.allclose(1, x):
            x = 0.999
        if not 0 < x < 1:
            raise ValueError('Invalid Beta: %s' % str(x))
        self.N += 1
        self.sum_log_x += log(x)
        self.sum_minus_log_x += log(1.-x)
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_log_x -= log(x)
        self.sum_minus_log_x -= log(1.-x)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if not 0 < x < 1:
            return -float('inf')
        return Beta.calc_predictive_logp(x, self.strength, self.balance)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        alpha = self.strength * self.balance
        beta = self.strength * (1. - self.balance)
        x = self.rng.beta(alpha, beta)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        data_logp = Beta.calc_log_likelihood(
            self.N, self.sum_log_x, self.sum_minus_log_x, self.strength,
            self.balance)
        prior_logp = Beta.calc_log_prior(
            self.strength, self.balance, self.mu, self.alpha, self.beta)
        return data_logp + prior_logp

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        n_samples = 100

        # Transition strength.
        def log_pdf_fun_str(strength):
            return (
                Beta.calc_log_likelihood(
                    self.N, self.sum_log_x, self.sum_minus_log_x,
                    strength, self.balance)
                + Beta.calc_log_prior(
                    strength, self.balance, self.mu, self.alpha, self.beta))

        self.strength = su.mh_sample(
            self.strength, log_pdf_fun_str, .5, [.0, float('Inf')],
            burn=n_samples, rng=self.rng)

        # Transition balance.
        def log_pdf_fun_bal(balance):
            return (
                Beta.calc_log_likelihood(
                    self.N, self.sum_log_x, self.sum_minus_log_x,
                    self.strength, balance)
                + Beta.calc_log_prior(
                    self.strength, balance, self.mu, self.alpha, self.beta))

        self.balance = su.mh_sample(
            self.balance, log_pdf_fun_bal, .25, [0, 1], burn=n_samples,
            rng=self.rng)

    def set_hypers(self, hypers):
        return

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
        return {}

    @staticmethod
    def name():
        return 'beta'

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
        log_strength = scipy.stats.expon.logpdf(strength, scale=mu)
        log_balance = scipy.stats.beta.logpdf(balance, alpha, beta)
        return log_strength + log_balance

    @staticmethod
    def sample_parameters(mu, alpha, beta, rng):
        strength = rng.exponential(scale=mu)
        balance = rng.beta(alpha, beta)
        return strength, balance
