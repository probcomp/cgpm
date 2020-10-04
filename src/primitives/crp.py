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
from collections import OrderedDict
from math import log

from scipy.special import gammaln

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu


class Crp(DistributionGpm):
    """Crp distribution over open set categoricals represented as integers.

    X[n] ~ Crp(\alpha | X[1],...,X[n-1])
    """

    def __init__(
            self, outputs, inputs,
            hypers=None, params=None, distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Distargs.
        self.N = 0
        self.data = OrderedDict()
        self.counts = OrderedDict()
        # Hyperparameters.
        if hypers is None: hypers = {}
        self.alpha = hypers.get('alpha', 1.)

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = int(observation[self.outputs[0]])
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

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        # Do not call DistributionGpm.logpdf since crp allows observed rowid.
        assert not inputs
        assert not constraints
        assert list(targets.keys()) == self.outputs
        x = int(targets[self.outputs[0]])
        if rowid in self.data:
            return 0 if self.data[rowid] == x else -float('inf')
        return Crp.calc_predictive_logp(x, self.N, self.counts, self.alpha)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            x = self.data[rowid]
        else:
            K = sorted(self.counts) + [max(self.counts) + 1] if self.counts\
                else [0]
            logps = [self.logpdf(rowid, {targets[0]: x}, None) for x in K]
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
        return {'alpha': self.alpha}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N': self.N, 'counts': self.counts}

    def get_distargs(self):
        return {}

    # Some Gibbs utils.

    def gibbs_logps(self, rowid, m=1):
        """Compute the CRP probabilities for a Gibbs transition of rowid,
        with table counts Nk, table assignments Z, and m auxiliary tables."""
        assert rowid in self.data
        assert 0 < m
        singleton = self.singleton(rowid)
        p_aux = self.alpha / float(m)
        p_rowid = p_aux if singleton else self.counts[self.data[rowid]]-1
        tables = self.gibbs_tables(rowid, m=m)
        def p_table(t):
            if t == self.data[rowid]: return p_rowid    # rowid table.
            if t not in self.counts: return p_aux       # auxiliary table.
            return self.counts[t]                       # regular table.
        return [log(p_table(t)) for t in tables]

    def gibbs_tables(self, rowid, m=1):
        """Retrieve a list of possible tables for rowid.

        If rowid is an existing customer, then the standard Gibbs proposal
        tables  are returned (i.e. with the rowid unincorporated). If
        rowid was a singleton table, then the table is re-used as a proposal
        and m-1 additional auxiliary tables are proposed, else m auxiliary
        tables are returned.

        If rowid is a new customer, then the returned tables are from the
        predictive distribution, (using m auxiliary tables always).
        """
        assert 0 < m
        K = sorted(self.counts)
        singleton = self.singleton(rowid)
        m_aux = m - 1 if singleton else m
        t_aux = [max(self.counts) + 1 + m for m in range(m_aux)]
        return K + t_aux

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
        # http://gershmanlab.webfactional.com/pubs/GershmanBlei12.pdf#page=4 (eq 8)
        return len(counts) * log(alpha) + sum(gammaln(list(counts.values()))) \
            + gammaln(alpha) - gammaln(N + alpha)
