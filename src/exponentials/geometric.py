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

from scipy.special import betaln

import gpmcc.utils.general as gu

from gpmcc.exponentials.distribution import DistributionGpm


class Geometric(DistributionGpm):
    """Geometric distribution data with beta prior on mu. Distirbution
    takes values x in 0,1,2,... where f(x) = p*(1-p)**x i.e. number of
    failures before the first success. Collapsed.

    mu ~ Beta(a, b)
    x ~ Geometric(mu)
    http://halweb.uc3m.es/esp/Personal/personas/mwiper/docencia/English/PhD_Bayesian_Statistics/ch3_2009.pdf
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

    def incorporate(self, rowid, query, evidence=None):
        DistributionGpm.incorporate(self, rowid, query, evidence)
        x = query[self.outputs[0]]
        if not (x % 1 == 0 and x >= 0):
            raise ValueError('Invalid Geometric: %s') % str(x)
        self.N += 1
        self.sum_x += x
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_x -= x

    def logpdf(self, rowid, query, evidence=None):
        DistributionGpm.logpdf(self, rowid, query, evidence)
        x = query[self.outputs[0]]
        if not (x % 1 == 0 and x >= 0):
            return -float('inf')
        return Geometric.calc_predictive_logp(
            x, self.N, self.sum_x, self.a, self.b)

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        DistributionGpm.simulate(self, rowid, query, evidence)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        an, bn = Geometric.posterior_hypers(self.N, self.sum_x, self.a, self.b)
        pn = self.rng.beta(an, bn)
        x = self.rng.geometric(pn) - 1
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Geometric.calc_logpdf_marginal(
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
