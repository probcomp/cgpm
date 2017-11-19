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

from scipy.special import gammaln

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu


class Exponential(DistributionGpm):
    """Exponential distribution with gamma prior on mu. Collapsed.

    mu ~ Gamma(a, b)
    x ~ Exponential(mu)
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Sufficient statistics.
        self.N = 0
        self.sum_x = 0
        # Hyperparameters.
        if hypers is None: hypers = {}
        self.a = hypers.get('a', 1)
        self.b = hypers.get('b', 1)
        assert self.a > 0
        assert self.b > 0

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if x < 0:
            raise ValueError('Invalid Exponential: %s' % str(x))
        self.N += 1
        self.sum_x += x
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_x -= x

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if x < 0:
            return -float('inf')
        return Exponential.calc_predictive_logp(
            x, self.N, self.sum_x, self.a, self.b)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        an, bn = Exponential.posterior_hypers(
            self.N, self.sum_x, self.a, self.b)
        mu = self.rng.gamma(an, scale=1./bn)
        x = self.rng.exponential(scale=1./mu)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Exponential.calc_logpdf_marginal(
            self.N, self.sum_x, self.a, self.b)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.b = hypers['b']
        self.a = hypers['a']

    def get_hypers(self):
        return {'a': self.a, 'b': self.b}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(.5, float(len(X)), n_grid)
        grids['b'] = gu.log_linspace(.5, float(len(X)), n_grid)
        return grids

    @staticmethod
    def name():
        return 'exponential'

    @staticmethod
    def is_collapsed():
        return True

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
    def calc_predictive_logp(x, N, sum_x, a, b):
        an,bn = Exponential.posterior_hypers(N, sum_x, a, b)
        am,bm = Exponential.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Exponential.calc_log_Z(an, bn)
        ZM = Exponential.calc_log_Z(am, bm)
        return  ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, a, b):
        an, bn = Exponential.posterior_hypers(N, sum_x, a, b)
        Z0 = Exponential.calc_log_Z(a, b)
        ZN = Exponential.calc_log_Z(an, bn)
        return ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + N
        bn = b + sum_x
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  gammaln(a) - a*log(b)
        return Z
