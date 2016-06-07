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

from scipy.special import gammaln
from scipy.stats import invgamma

import gpmcc.utils.config as cu
import gpmcc.utils.data as du
import gpmcc.utils.general as gu

from gpmcc.cgpm import CGpm


Data = namedtuple('Data', ['x', 'Y'])


class LinearRegression(CGpm):
    """Bayesian linear model with normal prior on regression parameters and
    inverse-gamma prior on both observation and regression variance.

    \sigma2 ~ Inverse-Gamma(a, b)
    w ~ Normal(\mu, \sigma2*I)
    y ~ Normal(x'w, \sigma2)
    """

    def __init__(self, outputs, inputs, hypers=None, params=None, distargs=None,
            rng=None):
        # io data.
        self.outputs = outputs
        self.inputs = inputs
        self.rng = gu.gen_rng() if rng is None else rng
        assert len(self.outputs) == 1
        assert len(self.inputs) >= 1
        assert self.outputs[0] not in self.inputs
        assert len(distargs['cctypes']) == len(self.inputs)
        # Input distargs.
        p, counts = zip(
            *[self._predictor_count(cctype, ccarg)
            for cctype, ccarg in zip(distargs['cctypes'], distargs['ccargs'])])
        self.inputs_discrete = {i:c for i, c in enumerate(counts) if c}
        self.p = sum(p)+1
        # Dataset.
        self.N = 0
        self.data = Data(x=OrderedDict(), Y=OrderedDict())
        # Hyper parameters.
        if hypers is None: hypers = {}
        self.a = hypers.get('a', 1.)
        self.b = hypers.get('b', 1.)
        self.mu = hypers.get('mu', np.zeros(self.p))
        self.V = hypers.get('V', np.eye(self.p))

    def incorporate(self, rowid, query, evidence):
        assert rowid not in self.data.x
        assert rowid not in self.data.Y
        x, y = self.preprocess(query, evidence)
        self.N += 1
        self.data.x[rowid] = x
        self.data.Y[rowid] = y

    def unincorporate(self, rowid):
        del self.data.x[rowid]
        del self.data.Y[rowid]
        self.N -= 1

    def logpdf(self, rowid, query, evidence):
        xt, yt = self.preprocess(query, evidence)
        assert rowid not in self.data.x
        return LinearRegression.calc_predictive_logp(
            xt, yt, self.N, self.data.Y.values(), self.data.x.values(), self.a,
            self.b, self.mu, self.V)

    def simulate(self, rowid, query, evidence, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        assert query == self.outputs
        if rowid in self.data.x:
            return self.data.x[rowid]
        xt, yt = self.preprocess(None, evidence)
        an, bn, mun, Vn_inv = LinearRegression.posterior_hypers(
            self.N, self.data.Y.values(), self.data.x.values(), self.a, self.b,
            self.mu, self.V)
        sigma2, b = LinearRegression.sample_parameters(
            an, bn, mun, np.linalg.inv(Vn_inv), self.rng)
        x = self.rng.normal(np.dot(yt, b), np.sqrt(sigma2))
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return LinearRegression.calc_logpdf_marginal(
            self.N, self.data.Y.values(), self.data.x.values(),
            self.a, self.b, self.mu, self.V)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0.
        assert hypers['b'] > 0.
        self.a = hypers['a']
        self.b = hypers['b']

    def get_hypers(self):
        return {'a': self.a, 'b':self.b}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {}

    def get_distargs(self):
        return {'inputs_discrete': self.inputs_discrete, 'p': self.p}

    @staticmethod
    def construct_hyper_grids(X, n_grid=300):
        grids = dict()
        # Plus 1 for single observation case.
        N = len(X) + 1.
        ssqdev = np.var(X) * len(X) + 1.
        # Data dependent heuristics.
        grids['a'] = gu.log_linspace(1./(10*N), 10*N, n_grid)
        grids['b'] = gu.log_linspace(ssqdev/100., ssqdev, n_grid)
        return grids

    @staticmethod
    def name():
        return 'linear_regression'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return True

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    def preprocess(self, query, evidence):
        distargs = self.get_distargs()
        x = query[self.outputs[0]] if query else query
        y = [evidence[c] for c in self.inputs]
        inputs_discrete, p = distargs['inputs_discrete'], distargs['p']
        y = du.dummy_code(y, inputs_discrete)
        if not set.issubset(set(self.inputs), set(evidence.keys())):
            raise TypeError(
                'LinearRegression requires inputs {}: {}'.format(
                    self.inputs, evidence.keys()))
        if self.outputs[0] in evidence:
            raise TypeError(
                'LinearRegression cannot condition on output {}: {}'.format(
                    self.outputs, evidence.keys()))
        if len(y) != p-1:
            raise TypeError(
                'LinearRegression requires input length {}: {}'.format(p, y))
        return x, [1] + y

    @staticmethod
    def _predictor_count(cct, cca):
        if cu.cctype_class(cct).is_numeric():
            p, counts = 1, None
        elif cca is not None and 'k' in cca: # XXX HACK
            p, counts = cca['k']-1, int(cca['k'])
        return int(p), counts

    @staticmethod
    def calc_predictive_logp(xs, ys, N, Y, x, a, b, mu, V):
        # Equation 19.
        an, bn, mun, Vn_inv = LinearRegression.posterior_hypers(
            N, Y, x, a, b, mu, V)
        am, bm, mum, Vm_inv = LinearRegression.posterior_hypers(
            N+1, Y+[ys], x+[xs], a, b, mu, V)
        ZN = LinearRegression.calc_log_Z(an, bn)
        ZM = LinearRegression.calc_log_Z(am, bm)
        return -(1/2.)*np.log(2*np.pi) + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, Y, x, a, b, mu, V):
        # Equation 19.
        an, bn, mun, Vn = LinearRegression.posterior_hypers(
            N, Y, x, a, b, mu, V)
        Z0 = LinearRegression.calc_log_Z(a, b)
        ZN = LinearRegression.calc_log_Z(an, bn)
        return -(N/2.)*np.log(2*np.pi) + ZN - Z0

    @staticmethod
    def posterior_hypers(N, Y, x, a, b, mu, V):
        if N == 0:
            assert len(x) == len(Y) == 0
            return a, b, mu, np.linalg.inv(V)
        # Equation 6.
        X, y = np.asarray(Y), np.asarray(x)
        assert X.shape == (N,len(mu))
        assert y.shape == (N,)
        V_inv = np.linalg.inv(V)
        XT = np.transpose(X)
        XTX = np.dot(XT, X)
        mun = np.dot(
            np.linalg.inv(V_inv + XTX),
            np.dot(V_inv, mu) + np.dot(XT, y))
        Vn_inv = V_inv + XTX
        an = a + N/2.
        bn = b + .5 * (
            np.dot(np.transpose(mu), np.dot(V_inv, mu))
            + np.dot(np.transpose(x), x)
            - np.dot(
                np.transpose(mun),
                np.dot(Vn_inv, mun)))
        return an, bn, mun, Vn_inv

    @staticmethod
    def calc_log_Z(a, b):
        return gammaln(a) - a*np.log(b)

    @staticmethod
    def sample_parameters(a, b, mu, V, rng):
        sigma2 = invgamma.rvs(a, scale=b, random_state=rng)
        b = rng.multivariate_normal(mu, sigma2 * V)
        return sigma2, b
