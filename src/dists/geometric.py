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
from scipy.special import betaln
from scipy.stats import beta, geom

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Geometric(DistributionGpm):
    """Geometric distribution data with beta prior on mu. Distirbution
    takes values x in 0,1,2,... where f(x) = p*(1-p)**x i.e. number of
    failures before the first success. Collapsed.

    mu ~ Beta(a, b)
    x ~ Geometric(mu)
    http://halweb.uc3m.es/esp/Personal/personas/mwiper/docencia/English/PhD_Bayesian_Statistics/ch3_2009.pdf
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
        assert float(x) == int(x) and 0 <= x
        self.N += 1.0
        self.sum_x += x

    def unincorporate(self, x, y=None):
        assert float(x) == int(x) and 0 <= x
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        self.sum_x -= x

    def logpdf(self, x, y=None):
        return Geometric.calc_predictive_logp(
            x, self.N, self.sum_x, self.a, self.b)

    def logpdf_marginal(self):
        return Geometric.calc_logpdf_marginal(
            self.N, self.sum_x, self.a, self.b)

    def simulate(self, y=None):
        an, bn = Geometric.posterior_hypers(
            self.N, self.sum_x, self.a, self.b)
        pn = self.rng.beta(an, bn)
        return self.rng.geometric(pn) - 1

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

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(1, float(len(X)) / 2., n_grid)
        grids['b'] = gu.log_linspace(.1, float(len(X)) / 2., n_grid)
        return grids

    @staticmethod
    def name():
        return 'geometric'

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
        if float(x) != int(x) or x < 0:
            return float('-inf')
        an, bn = Geometric.posterior_hypers(N, sum_x, a, b)
        am, bm = Geometric.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Geometric.calc_log_Z(an, bn)
        ZM = Geometric.calc_log_Z(am, bm)
        return  ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, a, b):
        an, bn = Geometric.posterior_hypers(N, sum_x, a, b)
        Z0 = Geometric.calc_log_Z(a, b)
        ZN = Geometric.calc_log_Z(an, bn)
        return ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + N
        bn = b + sum_x
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        return betaln(a, b)
