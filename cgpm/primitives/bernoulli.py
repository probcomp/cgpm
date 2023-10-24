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

from scipy.special import betaln

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu


class Bernoulli(DistributionGpm):
    """Bernoulli distribution with beta prior on bias theta.

    theta ~ Beta(alpha, beta)
    x ~ Bernoulli(theta)
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Sufficent statistics.
        self.N = 0
        self.x_sum = 0
        # Hyperparameters.
        if hypers is None: hypers = {}
        self.alpha = hypers.get('alpha', 1.)
        self.beta = hypers.get('beta', 1.)
        assert self.alpha > 0
        assert self.beta > 0

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if x not in [0, 1]:
            raise ValueError('Invalid Bernoulli: %s' % str(x))
        self.N += 1
        self.x_sum += x
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.x_sum -= x

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if x not in [0, 1]:
            return -float('inf')
        return Bernoulli.calc_predictive_logp(
            x, self.N, self.x_sum, self.alpha, self.beta)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        p0 = Bernoulli.calc_predictive_logp(
            0, self.N, self.x_sum, self.alpha, self.beta)
        p1 = Bernoulli.calc_predictive_logp(
            1, self.N, self.x_sum, self.alpha, self.beta)
        x = gu.log_pflip([p0, p1], rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Bernoulli.calc_logpdf_marginal(
            self.N, self.x_sum, self.alpha, self.beta)

    ##################
    # NON-GPM METHOD #
    ##################

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

    def get_distargs(self):
        return {'k': 2}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1., float(len(X)), n_grid)
        grids['beta'] = gu.log_linspace(1., float(len(X)),n_grid)
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
        log_denom = log(N + alpha + beta)
        if x == 1:
            return log(x_sum + alpha) - log_denom
        else:
            return log(N - x_sum + beta) - log_denom

    @staticmethod
    def calc_logpdf_marginal(N, x_sum, alpha, beta):
        return betaln(x_sum + alpha, N - x_sum + beta) - betaln(alpha, beta)
