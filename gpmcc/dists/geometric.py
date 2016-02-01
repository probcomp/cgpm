# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

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

    def __init__(self, N=0, sum_x=0, a=1, b=1, distargs=None):
        assert a > 0
        assert b > 0
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def incorporate(self, x):
        self.N += 1.0
        self.sum_x += x

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        self.sum_x -= x

    def logpdf(self, x):
        return Geometric.calc_predictive_logp(x, self.N, self.sum_x, self.a,
            self.b)

    def marginal_logp(self):
        return Geometric.calc_marginal_logp(self.N, self.sum_x, self.a,
            self.b)

    def singleton_logp(self, x):
        return Geometric.calc_predictive_logp(x, 0, 0, self.a, self.b)

    def simulate(self):
        an, bn = Geometric.posterior_hypers(self.N, self.sum_x, self.a,
            self.b)
        pn = beta.rvs(an, bn)
        return geom.rvs(pn) - 1

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.b = hypers['b']
        self.a = hypers['a']

    def get_hypers(self):
        return {
            'a': self.a,
            'b': self.b,
        }

    def get_suffstats(self):
        return {
            'N': self.N,
            'sum_x': self.sum_x,
        }

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
    def calc_marginal_logp(N, sum_x, a, b):
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
