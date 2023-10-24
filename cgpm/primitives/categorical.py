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

from builtins import range
from math import log
from collections import defaultdict

import numpy as np

from scipy.special import gammaln

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu


class Categorical(DistributionGpm):
    """Categorical distribution with symmetric dirichlet prior on
    category weight vector v.

    k := distarg
    v ~ Symmetric-Dirichlet(alpha/k)
    x ~ Categorical(v)
    https://web.archive.org/web/20170603230204/https://people.eecs.berkeley.edu/~stephentu/writeups/dirichlet-conjugate-prior.pdf
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Distargs.
        k = distargs.get('k', None)
        if k is None:
            raise ValueError('Categorical requires distarg `k`.')
        self.k = int(k)
        # Sufficient statistics.
        self.N = 0
        self.counts = np.zeros(self.k)
        # Hyperparameters.
        if hypers is None: 
            hypers = {}
            self.alpha = np.ones(self.k)
        else:
            assert len(hypers) == self.k
            self.alpha = np.array([
                hypers[f"alpha_{i}"]
                for i in range(self.k)])

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if not (x % 1 == 0 and 0 <= x < self.k):
            raise ValueError('Invalid Categorical(%d): %s' % (self.k, x))
        x = int(x)
        self.N += 1
        self.counts[x] += 1
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.counts[x] -= 1

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if not (x % 1 == 0 and 0 <= x < self.k):
            return -float('inf')
        return Categorical.calc_predictive_logp(
            int(x), self.N, self.counts, self.alpha)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}

        x = gu.pflip(self.counts + self.alpha, rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Categorical.calc_logpdf_marginal(self.N, self.counts, self.alpha)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        set(hypers.keys()) == set([[f'alpha_{i}'] for i in range(self.k)])
        for i in range(self.k):
            assert hypers[f'alpha_{i}'] > 0         
        self.alpha = np.array([
            hypers[f"alpha_{i}"]
            for i in range(self.k)])

    def get_hypers(self):
        return {f'alpha_{i}': self.alpha[i] for i in range(self.k)}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N' : self.N, 'counts' : list(self.counts)}

    def get_distargs(self):
        return {'k': self.k}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        # grid is a static method, so it can't have access to self.k
        # we'll circumvent this by making it a defaultdict
        grids = defaultdict(
            lambda: gu.log_linspace(1., float(len(X)), n_grid)

        )
        return grids

    @staticmethod
    def name():
        return 'categorical'

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
    def validate(x, K):
        return int(x) == float(x) and 0 <= x < K

    @staticmethod
    def calc_predictive_logp(x, N, counts, alpha):
        numer = log(alpha[x] + counts[x])
        denom = log(np.sum(counts + alpha))
        return numer - denom

    @staticmethod
    def calc_logpdf_marginal(N, counts, alpha):
        K = len(counts)
        A = sum(alpha)
        lg = sum(gammaln(counts[k] + alpha[k]) for k in range(K))
        return gammaln(A) - gammaln(A+N) + lg - sum(gammaln(alpha[k]) for k in range(K))
