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

from scipy.stats import multivariate_normal
from sklearn.decomposition import FactorAnalysis

from cgpm.cgpm import CGpm
from cgpm.mixtures.dim import Dim
from cgpm.utils import config as cu
from cgpm.utils import data as du
from cgpm.utils import general as gu


class LowDimensionalMvn(CGpm):
    """Factor analysis model with continuous latent variables z in a low
    dimensional space. The generative model for a vector x is

    z ~ Normal(0, I)    where z \in R^L.
    e ~ Normal(0, Psi)  where Psi = diag(v_1,...,v_D)
    x = W.z + mu + e    where W \in R^(DxL) and mu \in R^D, learning by EM.

    From standard results (Murphy Section 12.1)

        z ~ Normal(0, I)                Prior.

        x|z ~ Normal(W.z + mu, Psi)     Likelihood.

        x ~ Normal(mu, W.W'+Psi)        Marginal.

        z|x ~ Normal(m, S)              Posterior.
            S = inv(I + W'.inv(Psi).W)      (covariance)
            m = S(W'.inv(Psi).(x-mu))       (mean)

    The full joint distribution over [z,x] is then

    The mean of [z,x] is [0, mu]
    The covariance of [z,x] is (in block form)

        I           W'
      (LxL)       (LxD)

        W      W.W' + Psi
      (DxL)       (DxD)

    where the covariance W' is computed directly
    cov(z,x)    = cov(z, W.z + mu + e)
                = cov(z, W.z) + cov(z, mu) + cov(z, e)
                = cov(z, W.z)
                = cov(z,z).W'
                = W'

    Exercise: Confirm that expression for posterior z|x is consistent with
    conditioning directly on the joint [z,x] using Schur complement
    (Hint: see test suite).

    The latent variables are exposed as output variables, but may not be
    incorporated.
    """

    def __init__(self, outputs, inputs, hypers=None, params=None, distargs=None,
            rng=None):
        if params is None:
            params = {}
        if rng is None:
            rng = gu.gen_rng(1)
        # No inputs.
        if inputs:
            raise ValueError('FactorAnalysis rejects inputs: %s.' % inputs)
        # Find low dimensional space.
        L = distargs.get('L', None)
        if L is None:
            raise ValueError('Specify latent dimension L: %s.' % distargs)
        # Observable and latent variable indexes.
        D = len(outputs[:-L])
        if D < L:
            raise ValueError(
                'Latent dimension exceeds observed dimension: (%s,%s)'
                % (outputs[:-L], outputs[-L:]))
        # Build the object.
        self.rng = rng
        self.L = L
        self.D = D
        self.outputs = outputs
        self.obseravble = outputs[:-self.L]
        self.latents = outputs[-self.L:]
        self.mu = params.get('mu', np.zeros(D))
        self.Psi = params.get('Psi', np.eye(D))
        self.W = params.get('W', np.zeros((D,L)))
        self.data = OrderedDict()

    def incorporate(self, rowid, query, evidence=None):
        if rowid in self.data:
            raise ValueError('Already observed: %d.' % rowid)
        if evidence:
            raise ValueError('No evidence allowed: %s.' % evidence)
        if any(q in self.latents for q in query):
            raise ValueError('Cannot incorporate latents: %s.' % query)
        x = self.preprocess(query, evidence)
        self.N += 1
        self.data[rowid] = x

    def unincorporate(self, rowid):
        try:
            del self.data.x[rowid]
        except KeyError:
            raise ValueError('No such observation: %d' % rowid)
        self.N -= 1

    def logpdf(self, rowid, query, evidence=None):
        pass

    def simulate(self, rowid, query, evidence=None, N=None):
        pass

    def logpdf_score(self):
        pass

    ##################
    # NON-GPM METHOD #
    ##################

    def transition(self, N=None):
        X = np.asarray(self.data.values())
        fa = FactorAnalysis(n_components=self.L)
        fa.fit(X[~np.any(np.isnan(X), axis=1)])
        assert self.L, self.D == fa.components_.shape
        self.Psi = np.diag(fa.noise_variance_)
        self.mu = fa.mean_
        self.W = np.transpose(fa.components_)

    def get_params(self):
        return {
            'mu': self.mu,
            'Psi': self.Psi,
            'W': self.W
        }

    def get_distargs(self):
        return {'L': self.L}

    @staticmethod
    def name():
        return 'low_dimensional_mvn'

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    def preprocess(self, query, evidence):
        pass

    def reindex(self, query, reverse=False):
        func = lambda q: self.outputs[q] if reverse else self.outputs.index
        if isinstance(query, list):
            return [func(q) for q in query]
        else:
            return {func(q): query[q] for q in query}

    @staticmethod
    def mvn_marginalize(mu, cov, query, evidence):
        Q, E = query, evidence
        # Extract means.
        muQ = mu[Q]
        muE = mu[E]
        # Extract covariances.
        covQ = cov[Q][:,Q]
        covE = cov[E][:,E]
        covJ = cov[Q][:,E]
        covQE = np.row_stack((
            np.column_stack((covQ, covJ)),
            np.column_stack((covJ.T, covE))
        ))
        assert np.allclose(covQE, covQE.T)
        return muQ, muE, covQ, covE, covJ

    @staticmethod
    def mvn_condition(mu, cov, query, evidence):
        assert isinstance(query, list)
        assert isinstance(evidence, dict)
        assert len(mu) == cov.shape[0] == cov.shape[1]
        assert len(query) + len(evidence) <= len(mu)
        # Extract indexes and values from evidence.
        Ei, Ev = zip(*evidence.items())
        muQ, muE, covQ, covE, covJ = LowDimensionalMvn.mvn_marginalize(
            mu, cov, query, Ei)
        # Invoke Fact 4 from
        # http://web4.cs.ucl.ac.uk/staff/C.Bracegirdle/bayesTheoremForGaussians.pdf
        P = np.dot(covJ, np.linalg.inv(covE))
        muG = muQ + np.dot(P, Ev - muE)
        covG = covQ - np.dot(P, covJ.T)
        return muG, covG

    ####################
    # SERLIAZE METHODS #
    ####################

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = {self.data}
        metadata['distargs'] = self.get_distargs()
        metadata['params'] = self.get_params()
        metadata['factory'] = ('cgpm.pca.factor', 'LowDimensionalMvn')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        ldmvn = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            distargs=metadata['distargs'],
            params=metadata['params'],
            rng=rng)
        ldmvn.data = metadata['data']
        ldmvn.N = metadata['N']
        return ldmvn
