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

import sklearn.decomposition

from scipy.stats import multivariate_normal

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class FactorAnalysis(CGpm):
    """Factor analysis model with continuous latent variables z in a low
    dimensional space. The generative model for a vector x is

    z ~ Normal(0, I)    where z \in R^L.
    e ~ Normal(0, Psi)  where Psi = diag(v_1,...,v_D)
    x = W.z + mux + e    where W \in R^(DxL) and mux \in R^D, learning by EM.

    From standard results (Murphy Section 12.1)

        z ~ Normal(0, I)                Prior.

        x|z ~ Normal(W.z + mux, Psi)     Likelihood.

        x ~ Normal(mux, W.W'+Psi)        Marginal.

        z|x ~ Normal(m, S)              Posterior.
            S = inv(I + W'.inv(Psi).W)      (covariance)
            m = S(W'.inv(Psi).(x-mux))       (mean)

    The full joint distribution over [z,x] is then

    The mean of [z,x] is [0, mux]
    The covariance of [z,x] is (in block form)

        I           W'
      (LxL)       (LxD)

        W      W.W' + Psi
      (DxL)       (DxD)

    where the covariance W' is computed directly
    cov(z,x)    = cov(z, W.z + mux + e)
                = cov(z, W.z) + cov(z, mux) + cov(z, e)
                = cov(z, W.z)
                = cov(z,z).W'
                = W'

    Exercise: Confirm that expression for posterior z|x is consistent with
    conditioning directly on the joint [z,x] using Schur complement
    (Hint: see test suite).

    The latent variables are exposed as output variables, but may not be
    incorporated.
    """

    def __init__(self, outputs, inputs, params=None, distargs=None, rng=None):
        # Default parameter settings.
        if params is None:
            params = {}
        # Entropy.
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
        # Dimensions.
        self.L = L
        self.D = D
        # Varible indexes.
        self.outputs = outputs
        self.latents = outputs[-self.L:]
        # Dataset.
        self.data = OrderedDict()
        # Parameters of Factor Analysis.
        self.mux = params.get('mux', np.zeros(D))
        self.Psi = params.get('Psi', np.eye(D))
        self.W = params.get('W', np.zeros((D,L)))
        # Parameters of joint distribution [x,z].
        self.mu, self.cov = self.joint_parameters()

    def incorporate(self, rowid, query, evidence=None):
        # No duplicate observation.
        if rowid in self.data:
            raise ValueError('Already observed: %d.' % rowid)
        # No inputs.
        if evidence:
            raise ValueError('No evidence allowed: %s.' % evidence)
        # No unknown variables.
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))
        # No incorporation of latent variables.
        if any(q in self.latents for q in query):
            raise ValueError('Cannot incorporate latent vars: (%s,%s,%s).'
                % (query, self.outputs, self.latents))
        # Reindex the query variables.
        query_r = self.reindex(query)
        # Incorporate observed variables.
        x = [query_r.get(i, np.nan) for i in xrange(self.D)]
        # Update dataset and counts.
        self.data[rowid] = x
        self.N += 1

    def unincorporate(self, rowid):
        try:
            del self.data.x[rowid]
        except KeyError:
            raise ValueError('No such observation: %d.' % rowid)
        self.N -= 1

    def logpdf(self, rowid, query, evidence=None):
        # XXX Deal with observed rowids.
        if evidence is None:
            evidence = {}
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))
        if any(q in evidence for q in query):
            raise ValueError('Duplicate variable: (%s,%s).' % (query, evidence))
        # Reindex variables.
        query_r = self.reindex(query.keys)
        evidence_r = self.reindex(evidence)
        # Retrieve conditional distribution.
        muG, covG = FactorAnalysis.mvn_condition(
            self.mu, self.cov, query_r.keys(), evidence_r)
        # Compute log density.
        return multivariate_normal.logpdf(query_r.values(), mean=muG, cov=covG)

    def simulate(self, rowid, query, evidence=None, N=None):
        # XXX Deal with observed rowids.
        if evidence is None:
            evidence = {}
        if any(q in evidence for q in query):
            raise ValueError('Duplicate variable: (%s,%s).' % (query, evidence))
        # Reindex variables.
        query_r = self.reindex(query.keys())
        evidence_r = self.reindex(evidence)
        # Retrieve conditional distribution.
        muG, covG = FactorAnalysis.mvn_condition(
            self.mu, self.cov, query_r, evidence_r)
        # Generate samples.
        sample = multivariate_normal.rvs(
            mean=muG, cov=covG, size=N, random_state=self.rng)
        def get_sample(s):
            return dict(zip(query, s))
        return get_sample(sample) if N is None else map(get_sample, sample)

    def logpdf_score(self):
        def compute_logpdf(x):
            assert len(x) == self.D
            query = {i:v for i,v in enumerate(x) if not np.isnan(v)}
            return self.logpdf(-1, query, evidence=None)
        return sum(compute_logpdf(x) for x in self.data)

    def transition(self, N=None):
        X = np.asarray(self.data.values())
        # Only run inference on observations without missing entries.
        fa = sklearn.decomposition.FactorAnalysis(n_components=self.L)
        fa.fit(X[~np.any(np.isnan(X), axis=1)])
        assert self.L, self.D == fa.components_.shape
        # Update parameters of Factor Analysis.
        self.Psi = np.diag(fa.noise_variance_)
        self.mux = fa.mean_
        self.W = np.transpose(fa.components_)
        self.mu, self.cov = self.joint_parameters()

    # --------------------------------------------------------------------------
    # Internal.

    def get_params(self):
        return {
            'mu': self.mu,
            'Psi': self.Psi,
            'W': self.W
        }

    def get_distargs(self):
        return {
            'L': self.L
        }

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

    # --------------------------------------------------------------------------
    # Helper.

    def reindex(self, query, reverse=False):
        func = lambda q: self.outputs[q] if reverse else self.outputs.index
        if isinstance(query, list):
            return [func(q) for q in query]
        else:
            return {func(q): query[q] for q in query}

    def joint_parameters(self):
        mean = np.concatenate((np.zeros(self.L), self.mu))
        cov = np.row_stack((
            np.column_stack((np.eye(self.L), self.W.T)),
            np.column_stack((self.W, np.dot(self.W, self.W.T) + self.Psi))
        ))
        return mean, cov

    @staticmethod
    def mvn_marginalize(mu, cov, query, evidence):
        Q, E = query, evidence
        # Retrieve means.
        muQ = mu[Q]
        muE = mu[E]
        # Retrieve covariances.
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
        muQ, muE, covQ, covE, covJ = \
            FactorAnalysis.mvn_marginalize(mu, cov, query, Ei)
        # Invoke Fact 4 from, where G means given.
        # http://web4.cs.ucl.ac.uk/staff/C.Bracegirdle/bayesTheoremForGaussians.pdf
        P = np.dot(covJ, np.linalg.inv(covE))
        muG = muQ + np.dot(P, Ev - muE)
        covG = covQ - np.dot(P, covJ.T)
        return muG, covG

    # --------------------------------------------------------------------------
    # Serialization.

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = {self.data}
        metadata['distargs'] = self.get_distargs()
        metadata['params'] = self.get_params()
        metadata['factory'] = ('cgpm.pca.factor', 'FactorAnalysis')
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
