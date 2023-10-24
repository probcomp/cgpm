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

from cgpm.primitives.distribution import DistributionGpm
from cgpm.primitives.normal import Normal
from cgpm.utils import general as gu


class Lognormal(DistributionGpm):
    """Log-normal (zero-bounded) distribution with normal prior on mean and
    gamma prior on precision. Collapsed.

    rho ~ Gamma(a, b)
    mu ~ Normal(m, t)
    x ~ Lognormal(mu, rho)
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Sufficient statistics.
        self.N = 0
        self.sum_log_x_sq = 0
        self.sum_log_x = 0
        # Hyperparameters.
        if hypers is None: hypers = {}
        self.m = hypers.get('m', 1.)
        self.r = hypers.get('r', 1.)
        self.s = hypers.get('s', 1.)
        self.nu = hypers.get('nu', 1.)
        assert self.r > 0.
        assert self.s > 0.
        assert self.nu > 0.

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if x <= 0:
            raise ValueError('Invalid Lognormal: %s' % str(x))
        self.N += 1
        self.sum_log_x += log(x)
        self.sum_log_x_sq += log(x) * log(x)
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_log_x -= log(x)
        self.sum_log_x_sq -= log(x) * log(x)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if x <= 0:
            return -float('inf')
        return - log(x) + \
            Normal.calc_predictive_logp(
                log(x), self.N, self.sum_log_x, self.sum_log_x_sq, self.m,
                self.r, self.s, self.nu)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        # XXX This implementation is not verified but will be covered in
        # future univariate simulate tests, see Github issue #14.
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        # Simulate normal parameters.
        mn, rn, sn, nun = Normal.posterior_hypers(
            self.N, self.sum_log_x, self.sum_log_x_sq, self.m, self.r,
            self.s, self.nu)
        mu, rho = Normal.sample_parameters(mn, rn, sn, nun, self.rng)
        xn = self.rng.normal(loc=mu, scale=rho**-.5)
        x = np.exp(xn)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return -self.sum_log_x + \
            Normal.calc_logpdf_marginal(
                self.N, self.sum_log_x, self.sum_log_x_sq, self.m, self.r,
                self.s, self.nu)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['r'] > 0.
        assert hypers['s'] > 0.
        assert hypers['nu'] > 0.
        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def get_hypers(self):
        return {'m': self.m, 'r': self.r, 's': self.s, 'nu': self.nu}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N': self.N, 'sum_log_x': self.sum_log_x,
            'sum_log_x_sq': self.sum_log_x_sq}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['m'] = gu.log_linspace(1e-4, max(X), n_grid)
        grids['r'] = gu.log_linspace(.1, float(len(X)), n_grid)
        grids['nu'] = gu.log_linspace(.1, float(len(X)), n_grid)
        grids['s'] = gu.log_linspace(.1, float(len(X)), n_grid)
        return grids

    @staticmethod
    def name():
        return 'lognormal'

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

    @staticmethod
    def preprocess(x, y, distargs=None):
        if x <= 0:
            raise ValueError('Lognormal requires [0,inf): {}'.format(x))
        return x, y

