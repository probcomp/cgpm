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

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu
from cgpm.utils import mvnormal as multivariate_normal


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

    def __init__(self, outputs, inputs, L=None, distargs=None, params=None,
            rng=None):
        # Default parameter settings.
        if params is None:
            params = {}
        # Entropy.
        if rng is None:
            rng = gu.gen_rng(1)
        # No inputs.
        if inputs:
            raise ValueError('FactorAnalysis rejects inputs: %s.' % inputs)
        # Correct outputs.
        if len(outputs) < 2:
            raise ValueError('FactorAnalysis needs >= 2 outputs: %s.' % outputs)
        if len(set(outputs)) != len(outputs):
            raise ValueError('Duplicate outputs: %s.' % outputs)
        # Find low dimensional space.
        if L is None:
            raise ValueError('Specify latent dimension L: %s.' % L)
        if L == 0:
            raise ValueError('Latent dimension at least 1: %s.' % L)
        # Observable and latent variable indexes.
        D = len(outputs[:-L])
        if D < L:
            raise ValueError(
                'Latent dimension exceeds observed dimension: (%s,%s)'
                % (outputs[:-L], outputs[-L:]))
        # Parameters.
        mux = params.get('mux', np.zeros(D))
        Psi = params.get('Psi', np.eye(D))
        W = params.get('W', np.zeros((D,L)))
        # Build the object.
        self.rng = rng
        # Dimensions.
        self.L = L
        self.D = D
        # Varible indexes.
        self.outputs = outputs
        self.latents = outputs[-self.L:]
        self.inputs = []
        # Dataset.
        self.data = OrderedDict()
        self.N = 0
        # Parameters of Factor Analysis.
        self.mux = np.asarray(mux)
        self.Psi = np.asarray(Psi)
        self.W = np.asarray(W)
        # Parameters of joint distribution [x,z].
        self.mu, self.cov = self.joint_parameters()
        # Internal factor analysis model.
        self.fa = None

    def incorporate(self, rowid, query, evidence=None):
        # No duplicate observation.
        if rowid in self.data:
            raise ValueError('Already observed: %d.' % rowid)
        # No inputs.
        if evidence:
            raise ValueError('No evidence allowed: %s.' % evidence)
        if not query:
            raise ValueError('No query specified: %s.' % query)
        # No unknown variables.
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))
        # No incorporation of latent variables.
        if any(q in self.latents for q in query):
            raise ValueError('Cannot incorporate latent vars: (%s,%s,%s).'
                % (query, self.outputs, self.latents))
        # Reindex the query variables.
        query_r = {self.outputs.index(q): query[q] for q in query}
        # Incorporate observed variables.
        x = [query_r.get(i, np.nan) for i in xrange(self.D)]
        # Update dataset and counts.
        self.data[rowid] = x
        self.N += 1

    def unincorporate(self, rowid):
        try:
            del self.data[rowid]
        except KeyError:
            raise ValueError('No such observation: %d.' % rowid)
        self.N -= 1

    def logpdf(self, rowid, query, evidence=None):
        # XXX Deal with observed rowid.
        evidence = self.populate_evidence(rowid, query, evidence)
        if not query:
            raise ValueError('No query: %s.' % query)
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))
        if any(q in evidence for q in query):
            raise ValueError('Duplicate variable: (%s,%s).' % (query, evidence))
        # Reindex variables.
        query_r = self.reindex(query)
        evidence_r = self.reindex(evidence)
        # Retrieve conditional distribution.
        muG, covG = FactorAnalysis.mvn_condition(
            self.mu, self.cov, query_r.keys(), evidence_r)
        # Compute log density.
        x = np.array(query_r.values())
        return multivariate_normal.logpdf(x, muG, covG)

    def simulate(self, rowid, query, evidence=None, N=None):
        # XXX Deal with observed rowid.
        evidence = self.populate_evidence(rowid, query, evidence)
        if any(q in evidence for q in query):
            raise ValueError('Duplicate variable: (%s,%s).' % (query, evidence))
        # Reindex variables.
        query_r = self.reindex(query)
        evidence_r = self.reindex(evidence)
        # Retrieve conditional distribution.
        muG, covG = FactorAnalysis.mvn_condition(
            self.mu, self.cov, query_r, evidence_r)
        # Generate samples.
        sample = self.rng.multivariate_normal(mean=muG, cov=covG, size=N)
        def get_sample(s):
            return {query[0]:s} if isinstance(s, float) else dict(zip(query, s))
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
        self.fa = sklearn.decomposition.FactorAnalysis(n_components=self.L)
        self.fa.fit(X[~np.any(np.isnan(X), axis=1)])
        assert self.L, self.D == self.fa.components_.shape
        # Update parameters of Factor Analysis.
        self.Psi = np.diag(self.fa.noise_variance_)
        self.mux = self.fa.mean_
        self.W = np.transpose(self.fa.components_)
        self.mu, self.cov = self.joint_parameters()

    def populate_evidence(self, rowid, query, evidence):
        if evidence is None:
            evidence = {}
        if rowid in self.data:
            values = self.data[rowid]
            evidence_obs = {
                e:v for e,v in zip(self.outputs[:self.D], values)
                if not np.isnan(v) and e not in query and e not in evidence
            }
            evidence = gu.merged(evidence, evidence_obs)
        return evidence

    # --------------------------------------------------------------------------
    # Internal.

    def get_params(self):
        return {
            'mu': self.mu,
            'Psi': self.Psi,
            'W': self.W
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

    def reindex(self, query):
        # Reindex an output variable to its index in self.mu
        # self.mu has as the first L items the last L items of self.outputs
        # and as the remaining D items the first D items of self.outputs.
        # The following diagram is useful:
        # self.outputs:  12 14 -7 5 |  11 4  3
        #                <---D=4--->|<--L=3-->
        # raw indices:   0  1  2  3 |  4  5  6
        # reindexed:     3  4  5  6 |  0  1  2
        assert isinstance(query, (list, dict))
        def convert(q):
            i = self.outputs.index(q)
            return i - self.D if q in self.latents else i + self.L
        indexes = [convert(q) for q in query]
        if isinstance(query, list):
            return indexes
        else:
            return dict(zip(indexes, query.values()))

    def joint_parameters(self):
        mean = np.concatenate((np.zeros(self.L), self.mux))
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
        Ei, Ev = evidence.keys(), evidence.values()
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
        metadata['L'] = self.L
        metadata['data'] = self.data.items()

        # Store paramters as list for JSON.
        metadata['params'] = dict()
        metadata['params']['mux'] = self.mux.tolist()
        metadata['params']['Psi'] = self.Psi.tolist()
        metadata['params']['W'] = self.W.tolist()

        metadata['factory'] = ('cgpm.factor.factor', 'FactorAnalysis')
        return metadata


    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        fact = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            L=metadata['L'],
            params=metadata['params'],
            rng=rng)
        fact.data = OrderedDict(metadata['data'])
        fact.N = metadata['N']
        return fact
