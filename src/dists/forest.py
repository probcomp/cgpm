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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.misc import logsumexp

from gpmcc.utils import general as gu
from gpmcc.dists.distribution import DistributionGpm


class RandomForest(DistributionGpm):
    """RandomForest conditional distribution p(x|y) where x is categorical."""

    def __init__(self, alpha=.1, regressor=None, distargs=None, rng=None):
        self.rng = gu.gen_rng() if rng is None else rng
        # Number of categories.
        self.k = int(distargs['k'])
        # Number of conditions.
        self.p = len(distargs['cctypes'])
        # Outlier hyperparam.
        self.alpha = alpha
        # Sufficient statistics.
        self.N = 0
        self.x = []
        self.Y = []
        self.counts = np.zeros(self.k)
        # Random forest parameters.
        if regressor is None:
            regressor = RandomForestClassifier(random_state=self.rng)
        self.regressor = regressor

    def incorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        self.Y.append(y)
        self.x.append(x)
        self.counts[x] += 1
        self.N += 1

    def unincorporate(self, x, y=None):
        x, y = self.preprocess(x, y, self.get_distargs())
        for i in xrange(len(self.Y)):
            if np.allclose(self.Y[i], y) and self.x[i] == x:
                del self.x[i], self.Y[i]
                self.counts[x] -= 1
                self.N -= 1
                break
        else:
            raise ValueError('Observation %s not incorporated.' % str((x, y)))

    def logpdf(self, x, y=None):
        try: x, y = RandomForest.preprocess(x, y, self.get_distargs())
        except ValueError: return -float('inf')
        return RandomForest.calc_predictive_logp(
            x, y, self.regressor, self.counts, self.alpha)

    def logpdf_marginal(self):
        return RandomForest.calc_log_likelihood(
            self.x, self.Y, self.regressor, self.counts, self.alpha)

    def simulate(self, y=None):
        logps = [self.logpdf(x, y=y) for x in xrange(self.k)]
        return gu.log_pflip(logps, rng=self.rng)

    def transition_params(self):
        # Transition noise parameter.
        alphas = np.linspace(0.01, 0.99, 30)
        alpha_logps = [RandomForest.calc_log_likelihood(self.x, self.Y,
            self.regressor, self.counts, a) for a in alphas]
        index = gu.log_pflip(alpha_logps, rng=self.rng)
        self.alpha = alphas[index]
        # Transition forest.
        if len(self.Y) > 0:
            self.regressor.fit(self.Y, self.x)

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

    @staticmethod
    def preprocess(x, y, distargs=None):
        p, k = distargs['p'], distargs['k']
        if len(y) != p:
            raise TypeError(
                'RandomForest requires input length {}: {}'.format(p, y))
        if (int(x) != float(x)) or not (0 <= x < distargs['k']):
            raise ValueError(
                'RandomForest requires output in [0..{}): {}'.format(k, x))
        return int(x), y
