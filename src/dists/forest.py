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

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.misc import logsumexp

from gpmcc.utils import general as gu
from gpmcc.dists.distribution import DistributionGpm


class RandomForest(DistributionGpm):
    """RandomForest conditional distribution p(x|y) where x is categorical."""

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        self.rng = gu.gen_rng() if rng is None else rng
        self.outputs = outputs
        self.inputs = inputs
        self.rng = gu.gen_rng() if rng is None else rng
        assert len(self.outputs) == 1
        assert len(self.inputs) >= 1
        assert self.outputs[0] not in self.inputs
        assert len(distargs['cctypes']) == len(self.inputs)
        # Number of output categories and input dimension.
        self.k = int(distargs['k'])
        self.p = len(distargs['cctypes'])
        # Sufficient statistics.
        self.N = 0
        self.x = OrderedDict()
        self.Y = OrderedDict()
        self.counts = np.zeros(self.k)
        # Outlier and random forest parameters.
        if params is None: params = {}
        self.alpha = params.get('alpha', .1)
        self.regressor = params.get('regressor', None)
        if self.regressor is None:
            self.regressor = RandomForestClassifier(random_state=self.rng)

    def incorporate(self, rowid, query, evidence):
        assert rowid not in self.x
        assert rowid not in self.Y
        x, y = self.preprocess(query, evidence)
        self.N += 1
        self.counts[x] += 1
        self.x[rowid] = x
        self.Y[rowid] = y

    def unincorporate(self, rowid):
        del self.x[rowid]
        del self.Y[rowid]
        self.N -= 1

    def logpdf(self, rowid, query, evidence):
        assert query.keys() == self.outputs
        assert rowid not in self.x
        try:
            x, y = self.preprocess(query, evidence)
        except ValueError:
            return -float('inf')
        return RandomForest.calc_predictive_logp(
            x, y, self.regressor, self.counts, self.alpha)

    def simulate(self, rowid, query, evidence):
        assert query == self.outputs
        if rowid in self.x:
            return self.x[rowid]
        logps = [self.logpdf(rowid, {query[0]:x}, evidence)
            for x in xrange(self.k)]
        return gu.log_pflip(logps, rng=self.rng)

    def logpdf_score(self):
        return RandomForest.calc_log_likelihood(
            self.x.values(), self.Y.values(), self.regressor,
            self.counts, self.alpha)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        # Transition noise parameter.
        alphas = np.linspace(0.01, 0.99, 30)
        alpha_logps = [RandomForest.calc_log_likelihood(self.x, self.Y,
            self.regressor, self.counts, a) for a in alphas]
        index = gu.log_pflip(alpha_logps, rng=self.rng)
        self.alpha = alphas[index]
        # Transition forest.
        if len(self.Y) > 0:
            self.regressor.fit(self.Y.values(), self.x.values())

    def set_hypers(self, hypers):
        return

    def get_hypers(self):
        return {}

    def get_params(self):
        return {'forest': self.regressor, 'alpha': self.alpha}

    def get_suffstats(self):
        return {}

    def get_distargs(self):
        return {'k': self.k, 'p': self.p}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = np.linspace(0.01, 0.99, n_grid)
        return grids

    @staticmethod
    def name():
        return 'random_forest'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_conditional():
        return True

    @staticmethod
    def is_numeric():
        return False

    ##################
    # HELPER METHODS #
    ##################

    def preprocess(self, query, evidence):
        x = query[self.outputs[0]]
        y = [evidence[c] for c in sorted(evidence)]
        distargs = self.get_distargs()
        p, k = distargs['p'], distargs['k']
        if len(y) != p:
            raise TypeError(
                'RandomForest requires input length {}: {}'.format(p, y))
        if not (x % 1 == 0 and 0 <= x < distargs['k']):
            raise ValueError(
                'RandomForest requires output in [0..{}): {}'.format(k, x))
        return int(x), y

    @staticmethod
    def validate(x, K):
        return int(x) == float(x) and 0 <= x < K

    @staticmethod
    def calc_log_likelihood(X, Y, regressor, counts, alpha):
        return np.sum([RandomForest.calc_predictive_logp(
            x, y, regressor, counts, alpha) for x, y in zip(X,Y)])

    @staticmethod
    def calc_predictive_logp(x, y, regressor, counts, alpha):
        logp_uniform = -np.log(len(counts))
        if not hasattr(regressor, 'classes_'):
            return logp_uniform
        elif x not in regressor.classes_:
            return np.log(alpha) + logp_uniform
        else:
            index = list(regressor.classes_).index(x)
            logp_rf = regressor.predict_log_proba([y])[0][index]
            return logsumexp([
                np.log(alpha) + logp_uniform,
                np.log(1-alpha) + logp_rf])
