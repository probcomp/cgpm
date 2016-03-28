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

from gpmcc.utils import general as gu
from gpmcc.dists.distribution import DistributionGpm

class RandomForest(DistributionGpm):
    """RandomForest conditional distribution p(x|y) where x is categorical."""

    def __init__(self, distargs=None):
        self.k = int(distargs['k'])
        self.p = len(distargs['cctypes'])
        self.x = []
        self.Y = []
        self.counts = np.zeros(self.k)
        self.regressor = RandomForestClassifier(random_state=0)

    def bulk_incorporate(self, X, Y=None):
        # For effeciency.
        pass

    def bulk_unincorporate(self, X, Y=None):
        # For effeciency.
        pass

    def incorporate(self, x, y=None):
        assert len(y) == self.p
        assert x <= self.k
        self.Y.append(y)
        self.x.append(x)
        self.counts[x] += 1
        self.regressor.fit(self.Y, self.x)

    def unincorporate(self, x, y=None):
        assert len(y) == self.p
        for i in xrange(len(self.Y)):
            if np.allclose(self.Y[i], y) and self.x[i] == x:
                del self.x[i], self.Y[i]
                break
        else:
            raise ValueError('Observation %s not incorporated.' % str((x, y)))
        self.counts[x] -= 1
        if len(self.Y) > 0:
            self.regressor.fit(self.Y, self.x)

    def logpdf(self, x, y=None):
        if len(self.Y) == 0:
            return 0
        # Should this just be the definition of the predictive logpdf?
        if self.counts[x] == 0:
            current_logp = self.calc_log_likelihood(
                self.regressor, self.x, self.Y)
            temp_regressor = RandomForestClassifier(random_state=0)
            self.Y.append(y)
            self.x.append(x)
            temp_regressor.fit(self.Y, self.x)
            temp_logp = self.calc_log_likelihood(temp_regressor, self.x, self.Y)
            del self.x[-1], self.Y[-1]
            return temp_logp - current_logp
        else:
            return self.calc_predictive_logp(self.regressor, x, y)

    def logpdf_marginal(self):
        if len(self.Y) == 0:
            return 0
        return self.calc_log_likelihood(self.regressor, self.x, self.Y)

    def simulate(self, y=None):
        logps = self.regressor.predict_log_proba([y])[0]
        return gu.log_pflip(logps)

    def transition_params(self):
        self.regressor.fit(self.Y, self.x)

    def set_hypers(self, hypers):
        return

    def get_hypers(self):
        return {}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=20):
        return {}

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
    def calc_log_likelihood(regressor, X, Y):
        logps = regressor.predict_log_proba(Y)
        classes = list(regressor.classes_)
        lookup = {c:classes.index(c) for c in classes}
        return np.sum([logp[lookup[X[i]]] for i, logp in enumerate(logps)])

    @staticmethod
    def calc_predictive_logp(regressor, x, y):
        return regressor.predict_log_proba([y])[0][x]
