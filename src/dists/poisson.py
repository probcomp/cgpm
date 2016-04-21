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

class Poisson(DistributionGpm):
    """Poisson distribution with gamma prior on mu. Collapsed.

    mu ~ Gamma(a, b)
    x ~ Poisson(mu)
    """

    def __init__(self, N=0, sum_x=0, sum_log_fact_x=0, a=1, b=1,
            distargs=None, rng=None):
        assert a > 0
        assert b > 0
        self.rng = gu.gen_rng() if rng is None else rng
        # Sufficient statistics.
        self.sum_x = sum_x
        self.N = N
        self.sum_log_fact_x = sum_log_fact_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def incorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N += 1.0
        self.sum_x += x
        self.sum_log_fact_x += gammaln(x+1)

    def unincorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        self.sum_x -= x
        self.sum_log_fact_x -= gammaln(x+1)

    def logpdf(self, x, y=None):
        try: x, y = self.preprocess(x, y, self.get_distargs())
        except ValueError: return -float('inf')
        return Poisson.calc_predictive_logp(
            x, self.N, self.sum_x, self.a, self.b)

    def logpdf_marginal(self):
        return Poisson.calc_logpdf_marginal(
            self.N, self.sum_x, self.sum_log_fact_x, self.a, self.b)

    def simulate(self, y=None):
        an, bn = Poisson.posterior_hypers(
            self.N, self.sum_x, self.a, self.b)
        return self.rng.negative_binomial(an, bn/(bn+1.))

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.a = hypers['a']
        self.b = hypers['b']

    def get_hypers(self):
        return {'a': self.a, 'b': self.b}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N': self.N, 'sum_x' : self.sum_x,
            'sum_log_fact_x': self.sum_log_fact_x}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        # only use integers for a so we can nicely draw from a negative binomial
        # in predictive_draw
        grids['a'] = np.unique(np.round(np.linspace(1, len(X), n_grid)))
        grids['b'] = gu.log_linspace(.1, float(len(X)), n_grid)
        return grids

    @staticmethod
    def name():
        return 'poisson'

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
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, a, b):
        an, bn = Poisson.posterior_hypers(N, sum_x, a, b)
        am, bm = Poisson.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Poisson.calc_log_Z(an, bn)
        ZM = Poisson.calc_log_Z(am, bm)
        return  ZM - ZN - gammaln(x+1)

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, sum_log_fact_x, a, b):
        an, bn = Poisson.posterior_hypers(N, sum_x, a, b)
        Z0 = Poisson.calc_log_Z(a, b)
        ZN = Poisson.calc_log_Z(an, bn)
        return ZN - Z0 - sum_log_fact_x

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + sum_x
        bn = b + N
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  gammaln(a) - a*log(b)
        return Z

    @staticmethod
    def preprocess(x, y, distargs=None):
        if float(x) != int(x) or x < 0:
            raise ValueError('Poisson requires [0,1,..): {}'.format(x))
        return int(x), y
