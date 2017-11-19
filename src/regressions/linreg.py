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

from math import log
from math import pi
from math import sqrt

from collections import OrderedDict
from collections import namedtuple

import numpy as np

from numpy.linalg import det

from scipy.special import gammaln

from cgpm.cgpm import CGpm
from cgpm.mixtures.dim import Dim
from cgpm.utils import config as cu
from cgpm.utils import data as du
from cgpm.utils import general as gu


LOG2PI = log(2*pi)
Data = namedtuple('Data', ['x', 'Y'])


class LinearRegression(CGpm):
    """Bayesian linear model with normal prior on regression parameters and
    inverse-gamma prior on both observation and regression variance.

    Reference
    http://www.biostat.umn.edu/~ph7440/pubh7440/BayesianLinearModelGoryDetails.pdf


    Y_i = w' X_i + \sigma^2
        Response data                   Y_i \in R
        Covariate vector                X_i \in R^p
        Regression coefficients         w \in R^p
        Regression variance             \sigma^2 \in R

    Hyperparameters:                    a=1, b=1, V=I, mu=[0], dimension=p

    Parameters:                         \sigma2 ~ Inverse-Gamma(a, b)
                                        w ~ MVNormal(\mu, \sigma2*I)

    Data                                Y_i|x_i ~ Normal(w' x_i, \sigma2)
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
        assert len(distargs['inputs']['stattypes']) == len(self.inputs)
        self.input_cctypes = distargs['inputs']['stattypes']
        self.input_ccargs = distargs['inputs']['statargs']
        # Determine number of covariates (with 1 bias term) and number of
        # categories for categorical covariates.
        p, counts = zip(*[
            self._predictor_count(cctype, ccarg) for cctype, ccarg
            in zip(self.input_cctypes, self.input_ccargs)])
        self.p = sum(p)+1
        self.inputs_discrete = {i:c for i, c in enumerate(counts) if c}
        # For numerical covariates, map index in inputs to index in code.
        self.lookup_numerical_index = self.input_to_code_index()
        # Dataset.
        self.N = 0
        self.data = Data(x=OrderedDict(), Y=OrderedDict())
        # Hyper parameters.
        if hypers is None: hypers = {}
        self.a = hypers.get('a', 1.)
        self.b = hypers.get('b', 1.)
        self.mu = hypers.get('mu', np.zeros(self.p))
        self.V = hypers.get('V', np.eye(self.p))

    def incorporate(self, rowid, observation, inputs=None):
        assert rowid not in self.data.x
        assert rowid not in self.data.Y
        if self.outputs[0] not in observation:
            raise ValueError('No observation in incorporate: %s' % observation)
        x, y = self.preprocess(observation, inputs)
        self.N += 1
        self.data.x[rowid] = x
        self.data.Y[rowid] = y

    def unincorporate(self, rowid):
        try:
            del self.data.x[rowid]
            del self.data.Y[rowid]
        except KeyError:
            raise ValueError('No such observation: %d' % rowid)
        self.N -= 1

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert rowid not in self.data.x
        assert not constraints
        xt, yt = self.preprocess(targets, inputs)
        return LinearRegression.calc_predictive_logp(
            xt, yt, self.N, self.data.Y.values(), self.data.x.values(), self.a,
            self.b, self.mu, self.V)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets == self.outputs
        assert not constraints
        if rowid in self.data.x:
            return {self.outputs[0]: self.data.x[rowid]}
        xt, yt = self.preprocess(None, inputs)
        sigma2, b = self.simulate_params()
        x = self.rng.normal(np.dot(yt, b), np.sqrt(sigma2))
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return LinearRegression.calc_logpdf_marginal(
            self.N, self.data.Y.values(), self.data.x.values(),
            self.a, self.b, self.mu, self.V)

    def simulate_params(self):
        an, bn, mun, Vn_inv = LinearRegression.posterior_hypers(
            self.N, self.data.Y.values(), self.data.x.values(), self.a, self.b,
            self.mu, self.V)
        return LinearRegression.sample_parameters(
            an, bn, mun, np.linalg.inv(Vn_inv), self.rng)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition(self, N=None):
        self.transition_hypers(N=N)

    def transition_hypers(self, N=None):
        if N is None:
            N = 1
        dim = Dim(
            self.outputs, [-10**8]+self.inputs,
            cctype=self.name(), hypers=self.get_hypers(),
            distargs=self.get_distargs(), rng=self.rng)
        dim.clusters[0] = self
        dim.transition_hyper_grids(X=self.data.x.values())
        for i in xrange(N):
            dim.transition_hypers()

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
        return {
            'inputs': {
                'stattypes': self.input_cctypes,
                'statargs': self.input_ccargs,
            } ,
            'inputs_discrete': self.inputs_discrete,
            'p': self.p,
        }

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

    def input_to_code_index(self):
        # Convert the index of a numerical variable in self.input to its index
        # in the dummy code. For instance, if inputs = [1,2,4] and 1 is
        # categorical with 3 terms, and 2 and 4 are numerical, the code is:
        # [bias, x1-0, x1-1, x1-3, x1-4, 2, 4]
        def compute_offset(i):
            before = [c for c in self.inputs[:i] if c in self.inputs_discrete]
            offset = sum(self.inputs_discrete[b]-1 for b in before)
            return 1+offset+i # 1 for the bias term.
        avoid = [self.inputs[i] for i in self.inputs_discrete]
        numericals = [c for c,i in enumerate(self.inputs) if i not in avoid]
        return {c: compute_offset(c) for c in numericals}


    def preprocess(self, targets, inputs):
        # Retrieve the value x of the target variable.
        if self.outputs[0] in inputs:
            raise ValueError('Cannot specify output as input %s: %s'
                % (self.outputs, inputs.keys()))
        if targets:
            x = targets.get(self.outputs[0], 'missing')
            if x == 'missing':
                raise ValueError('No targets: %s, %s'
                    % (self.outputs, targets))
            elif x is None or np.isnan(x):
                raise ValueError('Invalid targets: %s' % (targets,))
        else:
            x = None
        # Crash on missing inputs since it violates a CGPM contract!
        if set(inputs.keys()) != set(self.inputs):
            raise ValueError('Missing inputs: %s, %s' % (inputs, self.inputs))
        # Retrieve the covariates.
        y = [inputs.get(i) for i in self.inputs]
        # Dummy code covariates.
        y = du.dummy_code(y, self.inputs_discrete)
        assert len(y) == self.p-1
        return x, [1] + y

    @staticmethod
    def _predictor_count(cct, cca):
        # XXX Determine statistical types and arguments of inputs.
        if cct == 'numerical' or cu.cctype_class(cct).is_numeric():
            p, counts = 1, None
        elif cca is not None and 'k' in cca:
            # In dummy coding, if the category has values {1,...,K} then its
            # code contains (K-1) entries, where all zeros indicates value K.
            # However, we are going to treat all zeros indicating the input to
            # be a "wildcard" category, so that the code has K entries. This
            # way the queries are robust to unspecified or misspecified
            # categories.
            p, counts = cca['k'], int(cca['k'])+1
        return int(p), counts

    @staticmethod
    def calc_predictive_logp(xs, ys, N, Y, x, a, b, mu, V):
        # Equation 19.
        an, bn, _mun, Vn_inv = LinearRegression.posterior_hypers(
            N, Y, x, a, b, mu, V)
        am, bm, _mum, Vm_inv = LinearRegression.posterior_hypers(
            N+1, Y+[ys], x+[xs], a, b, mu, V)
        ZN = LinearRegression.calc_log_Z(an, bn, Vn_inv)
        ZM = LinearRegression.calc_log_Z(am, bm, Vm_inv)
        return (-1/2.)*LOG2PI + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, Y, x, a, b, mu, V):
        # Equation 19.
        an, bn, _mun, Vn_inv = LinearRegression.posterior_hypers(
            N, Y, x, a, b, mu, V)
        Z0 = LinearRegression.calc_log_Z(a, b, np.linalg.inv(V))
        ZN = LinearRegression.calc_log_Z(an, bn, Vn_inv)
        return (-N/2.)*LOG2PI + ZN - Z0

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
    def calc_log_Z(a, b, V_inv):
        # Equation 19.
        return gammaln(a) + log(sqrt(1./det(V_inv))) - a * np.log(b)

    @staticmethod
    def sample_parameters(a, b, mu, V, rng):
        sigma2 = 1./rng.gamma(a, scale=1./b)
        b = rng.multivariate_normal(mu, sigma2 * V)
        return sigma2, b


    ####################
    # SERLIAZE METHODS #
    ####################

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = {'x': self.data.x, 'Y': self.data.Y}
        metadata['distargs'] = self.get_distargs()
        metadata['hypers'] = self.get_hypers()
        metadata['factory'] = ('cgpm.regressions.linreg', 'LinearRegression')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None: rng = gu.gen_rng(0)
        linreg = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            hypers=metadata['hypers'],
            distargs=metadata['distargs'],
            rng=rng)
        # json keys are strings -- convert back to integers.
        x = ((int(k), v) for k, v in metadata['data']['x'].iteritems())
        Y = ((int(k), v) for k, v in metadata['data']['Y'].iteritems())
        linreg.data = Data(x=OrderedDict(x), Y=OrderedDict(Y))
        linreg.N = metadata['N']
        return linreg
