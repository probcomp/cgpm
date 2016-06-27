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
from collections import namedtuple

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


Data = namedtuple('Data', ['x', 'Y'])


class RandomForest(CGpm):
    """RandomForest conditional GPM over k variables, with uniform noise model.

    p(x|Y,D) = \alpha*(1/k) + (1-\alpha)*RF(x|Y,D)
    """

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
        self.data = Data(x=OrderedDict(), Y=OrderedDict())
        self.counts = np.zeros(self.k)
        # Outlier and random forest parameters.
        if params is None: params = {}
        self.alpha = params.get('alpha', .1)
        self.regressor = params.get('regressor', None)
        if self.regressor is None:
            self.regressor = RandomForestClassifier(random_state=self.rng)

    def incorporate(self, rowid, query, evidence=None):
        assert rowid not in self.data.x
        assert rowid not in self.data.Y
        x, y = self.preprocess(query, evidence)
        self.N += 1
        self.counts[x] += 1
        self.data.x[rowid] = x
        self.data.Y[rowid] = y

    def unincorporate(self, rowid):
        try:
            del self.data.x[rowid]
            del self.data.Y[rowid]
        except KeyError:
            raise ValueError('No such observation: %d' % rowid)
        self.N -= 1

    def logpdf(self, rowid, query, evidence=None):
        assert query.keys() == self.outputs
        assert rowid not in self.data.x
        try:
            x, y = self.preprocess(query, evidence)
        except IndexError:
            return -float('inf')
        return RandomForest.calc_predictive_logp(
            x, y, self.regressor, self.counts, self.alpha)

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        assert query == self.outputs
        if rowid in self.data.x:
            return {self.outputs[0]: self.data.x[rowid]}
        logps = [self.logpdf(rowid, {query[0]: x}, evidence)
            for x in xrange(self.k)]
        x = gu.log_pflip(logps, rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return RandomForest.calc_log_likelihood(
            self.data.x.values(), self.data.Y.values(), self.regressor,
            self.counts, self.alpha)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        # Transition noise parameter.
        alphas = np.linspace(0.01, 0.99, 30)
        alpha_logps = [
            RandomForest.calc_log_likelihood(
                self.data.x.values(), self.data.Y.values(), self.regressor,
                self.counts, a)
            for a in alphas]
        self.alpha = gu.log_pflip(alpha_logps, array=alphas, rng=self.rng)
        # Transition forest.
        if len(self.data.Y) > 0:
            self.regressor.fit(self.data.Y.values(), self.data.x.values())

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
        # Retrieve the value x of the query variable.
        if self.outputs[0] in evidence:
            raise ValueError('Cannot condition on output {}: {}'.format(
                    self.outputs, evidence.keys()))
        x = query.get(self.outputs[0], None)
        if x is None or np.isnan(x):
            raise ValueError('Invalid query: %s.' % query)
        # Retrieve the evidence values.
        if not set.issubset(set(self.inputs), set(evidence.keys())):
            raise ValueError(
                'RandomForest requires inputs {}: {}'.format(
                    self.inputs, evidence.keys()))
        y = [evidence[c] for c in sorted(evidence)]
        if len(y) != self.p:
            raise ValueError(
                'RandomForest requires input length {}: {}'.format(self.p, y))
        if not (x % 1 == 0 and 0 <= x < self.k):
            raise IndexError(
                'RandomForest category not in [0..{}): {}'.format(self.k, x))
        return int(x), y

    @staticmethod
    def calc_log_likelihood(X, Y, regressor, counts, alpha):
        return sum(
            RandomForest.calc_predictive_logp(x, y, regressor, counts, alpha)
            for x, y in zip(X,Y))

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
            return gu.logsumexp([
                np.log(alpha) + logp_uniform,
                np.log(1-alpha) + logp_rf])
