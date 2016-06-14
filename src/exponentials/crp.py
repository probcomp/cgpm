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

from collections import OrderedDict
from math import log

from scipy.special import gammaln

import gpmcc.utils.general as gu

from gpmcc.exponentials.distribution import DistributionGpm


class Crp(DistributionGpm):
    """Crp distribution over open set categoricals represented as integers.

    X[n] ~ Crp(\alpha | X[1],...,X[n-1])
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Distargs.
        self.N = 0
        self.data = OrderedDict()
        self.counts = OrderedDict()
        # Hyperparameters.
        if hypers is None: hypers = {}
        self.alpha = hypers.get('alpha', 1.)

    def incorporate(self, rowid, query, evidence):
        DistributionGpm.incorporate(self, rowid, query, evidence)
        x = int(query[self.outputs[0]])
        self.N += 1
        if x not in self.counts:
            self.counts[x] = 0
        self.counts[x] += 1
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.counts[x] -= 1
        if self.counts[x] == 0:
            del self.counts[x]

    def logpdf(self, rowid, query, evidence):
        DistributionGpm.logpdf(self, rowid, query, evidence)
        x = int(query[self.outputs[0]])
        if rowid in self.data:
            return 0 if self.data[rowid] == x else -float('inf')
        return Crp.calc_predictive_logp(x, self.N, self.counts, self.alpha)

    def simulate(self, rowid, query, evidence, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        DistributionGpm.simulate(self, rowid, query, evidence)
        if rowid in self.data:
            x = self.data[rowid]
        else:
            K = sorted(self.counts) + [max(self.counts) + 1] if self.counts\
                else [0]
            logps = [self.logpdf(rowid, {query[0]: x}, evidence) for x in K]
            x = gu.log_pflip(logps, array=K, rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Crp.calc_logpdf_marginal(self.N, self.counts, self.alpha)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def get_hypers(self):
        return {}

    def get_params(self):
        return {'alpha': self.alpha}

    def get_suffstats(self):
        return {'N': self.N, 'counts': list(self.counts)}

    def get_distargs(self):
        return {}

    def gibbs_logps(self, rowid):
        """Compute the CRP probabilities for a Gibbs transition of rowid,
        with table counts Nk, table assignments Z, and m auxiliary tables."""
        assert rowid in self.data
        K = sorted(self.counts)
        singleton = self.singleton(rowid)
        t_aux = 0 if singleton else 1
        p_aux = self.alpha
        p_current = p_aux if singleton else self.counts[self.data[rowid]]-1
        def p_table(t):
            return p_current if t == self.data[rowid] else self.counts[t]
        return [log(p_table(t)) for t in K] + [log(p_aux)] * t_aux

    def gibbs_tables(self, rowid):
        t_aux = [] if self.singleton(rowid) else [max(self.counts)+1]
        return sorted(self.counts) + t_aux

    def singleton(self, rowid):
        return self.counts[self.data[rowid]] == 1 if rowid in self.data else 0

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1./len(X), len(X), n_grid)
        return grids

    @staticmethod
    def name():
        return 'crp'

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
    def calc_predictive_logp(x, N, counts, alpha):
        numerator = counts.get(x, alpha)
        denominator = N + alpha
        return log(numerator) - log(denominator)

    @staticmethod
    def calc_logpdf_marginal(N, counts, alpha):
        return len(counts) * log(alpha) + sum(gammaln(counts.values())) \
            + gammaln(alpha) - gammaln(N + alpha)
