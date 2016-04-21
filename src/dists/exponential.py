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
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import expon, gamma

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Exponential(DistributionGpm):
    """Exponential distribution with gamma prior on mu. Collapsed.

    mu ~ Gamma(a, b)
    x ~ Exponential(mu)
    """

    def __init__(self, N=0, sum_x=0, a=1, b=1, distargs=None, rng=None):
        assert a > 0
        assert b > 0
        self.rng = gu.gen_rng() if rng is None else rng
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def incorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N += 1.0
        self.sum_x += x

    def unincorporate(self, x, y=None):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        x, y = self.preprocess(x, y, self.get_distargs())
        self.N -= 1.0
        self.sum_x -= x

    def logpdf(self, x, y=None):
        try: x, y = self.preprocess(x, y, self.get_distargs())
        except ValueError: return -float('inf')
        return Exponential.calc_predictive_logp(
            x, self.N, self.sum_x, self.a, self.b)

    def logpdf_marginal(self):
        return Exponential.calc_logpdf_marginal(
            self.N, self.sum_x, self.a, self.b)

    def simulate(self, y=None):
        an, bn = Exponential.posterior_hypers(self.N, self.sum_x, self.a, self.b)
        mu = self.rng.gamma(an, scale=1./bn)
        return self.rng.exponential(scale=1./mu)

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

    @staticmethod
    def preprocess(x, y, distargs=None):
        if x < 0:
            raise ValueError('Exponential requires [0,inf): {}'.format(x))
        return x, y
