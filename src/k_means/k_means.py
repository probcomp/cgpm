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

from venture.lite.mvnormal import logpdf as mvn_logpdf

class KMeans(CGpm):
    """ K Means

    A heuristic clustering CGMP that is equivalent to a Gaussian mixture model
    where all covariance are diagonal, plus a K-means inspired transition
    operator.
    """

    def __init__(self, outputs, inputs, K=None, distargs=None, params=None,
            rng=None):
        # TODO: Decide whether or not to delete distargs and params.
        # Default parameter settings.
        if params is None:
            params = {}
        if distargs is None:
            distargs = {}
        # Entropy.
        if rng is None:
            rng = gu.gen_rng(1)
        # No inputs.
        if inputs:
            raise ValueError('KMeans rejects inputs: %s.' % inputs)
        if len(set(outputs)) != len(outputs):
            raise ValueError('Duplicate outputs: %s.' % outputs)
        # Find K clusters
        if K is None:
            raise ValueError('Specify number of clusters: %s.' % K)
        if K == 0:
            raise ValueError('Number of clusters is at least 1: %s.' % K)
        if 'outputs' in distargs and any(s != 'numerical'
                for s in distargs['outputs']['stattypes']):
            raise ValueError('Kmeans non-numerical outputs: %s.' % distargs)
        # Parameters.
        D = len(outputs)
        cluster_centers = params.get(
            'cluster_centers', [np.zeros(D) for _ in range(K)]
        )
        # Sigma parameter for cluster variances, as in sigma * I.
        cluster_sigmas = params.get(
            'cluster_sigmas', [1 for _ in range(K)]
        )
        # Mixing coefficients for the cluster
        mixing_coefficients = params.get(
            'mixing_coefficients', [1./K for _ in range(K)]
        )

        # Build the object.
        self.rng = rng
        # Dimensions.
        self.K = K
        self.D = D
        # Varible indexes.
        self.outputs = outputs
        self.inputs = []
        # Dataset.
        self.data = OrderedDict()
        self.N = 0
        # Parameters of K-means
        self.cluster_centers = cluster_centers
        self.cluster_sigmas = cluster_sigmas
        self.mixing_coefficients = mixing_coefficients

    def incorporate(self, rowid, query, evidence=None):
        # Copypasta from factor.py.
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
        # Incorporate observed variables.
        x = [query.get(i, np.nan) for i in xrange(self.D)]
        # Update dataset and counts.
        self.data[rowid] = x
        self.N += 1

    def unincorporate(self, rowid):
        try:
            del self.data[rowid]
        except KeyError:
            raise ValueError('No such observation: %d.' % rowid)
        self.N -= 1

    def get_logp_cluster(self, query, k, mixing_coefficient):
        """Get what each cluster contributes the the logpdf."""
        X = np.array(query.values())
        mu = np.array(
            [self.cluster_centers[k][index] for index in query.keys()]
        )
        Sigma = np.diag([self.cluster_sigmas[k]] * len(query))
        return\
            np.log(mixing_coefficient) + mvn_logpdf(X, mu, Sigma)

    def logpdf(self, rowid, query, evidence=None):
        # If the rowid is not None, populate the evidence with non-missing
        # values in of said row.
        evidence = self.populate_evidence(rowid, query, evidence)
        # Defensive programming.
        # Taken from factor.py.
        if not query:
            raise ValueError('No query: %s.' % query)
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))
        if evidence is not None and any(q in evidence for q in query):
            raise ValueError('Duplicate variable: (%s,%s).' % (query, evidence))
        X = np.array(query.values())
        # following eq. 4 and 5 from here:
        # http://bengio.abracadoudou.com/cv/publications/pdf/rr02-12.pdf
        if evidence:
            # if there is evidence, we need to re-wait the mixing coefficients.
            W = [self.get_logp_cluster(evidence, k, self.mixing_coefficients[k]) for k in range(self.K)]
            W_denominator = gu.logsumexp(W)
            W = [np.exp(w - W_denominator) for w in W]
        else:
            W = self.mixing_coefficients

        return gu.logsumexp([self.get_logp_cluster(query,k, W[k]) for k in range(self.K)])

    def simulate(self, rowid, query, evidence=None, N=None):
        evidence = self.populate_evidence(rowid, query, evidence)
        # following eq. 4 and 5 from here:
        # http://bengio.abracadoudou.com/cv/publications/pdf/rr02-12.pdf
        if evidence:
            # if there is evidence, we need to re-wait the mixing coefficients.
            W = [self.get_logp_cluster(evidence, k, self.mixing_coefficients[k]) for k in range(self.K)]
            W_denominator = gu.logsumexp(W)
            W = [np.exp(w - W_denominator) for w in W]
        else:
            W = self.mixing_coefficients
        # First, sample a cluster according to the weight vector W:
        def get_single_sample(k):
            mu = np.array(
                [self.cluster_centers[k][index] for index in query]
            )
            Sigma = np.diag([self.cluster_sigmas[k]] * len(query))
            sample_vector = self.rng.multivariate_normal(mean=mu, cov=Sigma, size=1)
            return {column:sample_vector[0][column] for column in query}

        return [get_single_sample(gu.pflip(W)) for _ in range(N)]


    def transition(self, N=None):
        raise NotImplementedError

    def populate_evidence(self, rowid, query, evidence):
        if evidence is None:
            evidence = {}
        if rowid in self.data:
            values = self.data[rowid]
            assert len(values) == len(self.outputs[:self.D])
            evidence_obs = {e:v for e,v in zip(self.outputs[:self.D], values)
                if not np.isnan(v) and e not in query and e not in evidence
            }
            evidence = gu.merged(evidence, evidence_obs)
        return evidence

