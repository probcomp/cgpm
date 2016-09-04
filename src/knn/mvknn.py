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

from scipy.stats import norm
from sklearn.neighbors import KDTree

from cgpm.cgpm import CGpm
from cgpm.utils import data as du
from cgpm.utils import general as gu

LocalGpm = namedtuple('LocalGpm', ['simulate', 'logpdf'])


class MultivariateKnn(CGpm):
    """Multivariate K-Nearest-Neighbors builds ML models on a per-query basis.

    TODO: Migrate description from Github Issue #128.
    """

    def __init__(self, outputs, inputs, K=None, M=None, distargs=None,
            params=None, rng=None):
        # Input validation.
        self._validate_init(outputs, inputs, K, M, distargs, params, rng)
        # Default arguments.
        if params is None:
            params = {}
        if rng is None:
            rng = gu.gen_rng(1)
        if M is None:
            M = K
        # Build the object.
        self.rng = rng
        # Varible indexes.
        self.outputs = outputs
        self.inputs = []
        # Distargs.
        self.stattypes = distargs['outputs']['stattypes']
        self.statargs = distargs['outputs']['statargs']
        self.levels = {
            o: self.statargs[i]['k']
            for i, o in enumerate(outputs) if self.stattypes[i] != 'numerical'
        }
        # Dataset.
        self.data = OrderedDict()
        self.N = 0
        # Ordering of the chain.
        self.ordering = list(self.rng.permutation(self.outputs))
        # Number of nearest neighbors.
        self.K = K
        self.M = M

    def incorporate(self, rowid, query, evidence=None):
        self._validate_incorporate(rowid, query, evidence)
        # Incorporate observed variables.
        x = [query.get(q, np.nan) for q in self.outputs]
        # Update dataset and counts.
        self.data[rowid] = x
        self.N += 1

    def unincorporate(self, rowid):
        self._validate_unincorporate(rowid)
        del self.data[rowid]
        self.N -= 1

    def logpdf(self, rowid, query, evidence=None):
        evidence = self._populate_evidence(rowid, query, evidence)
        # XXX Disable logpdf queries without evidence.
        if not evidence:
            raise ValueError('Provide at least one evidence: %s.' % evidence)
        self._validate_simulate_logpdf(rowid, query, evidence)
        # Retrieve the dataset and neighborhoods.
        dataset, neighborhoods = self._find_neighborhoods(query, evidence)
        models = [self._create_local_model_joint(query, dataset[n])
            for n in neighborhoods]
        # Compute logpdf in each neighborhood and simple average.
        lp = [m.logpdf(query) for m in models]
        return gu.logsumexp(lp) - np.log(len(models))

    def simulate(self, rowid, query, evidence=None, N=None):
        samples = 1 if N is None else N
        evidence = self._populate_evidence(rowid, query, evidence)
        self._validate_simulate_logpdf(rowid, query, evidence, samples)
        if evidence:
            # Retrieve the dataset and neighborhoods.
            dataset, neighborhoods = self._find_neighborhoods(query, evidence)
            models = [self._create_local_model_joint(query, dataset[n])
                for n in neighborhoods]
            # Sample the models.
            indices = self.rng.choice(len(models), size=samples)
            # Sample from each model.
            sampled_models = [models[i] for i in indices]
            results = [m.simulate(query) for m in sampled_models]
        else:
            results = self._simulate_recusrive(rowid, query, samples)
            assert len(results) == samples
        return results[0] if N is None else results

    def _simulate_recusrive(self, rowid, query, samples):
        # Fallback if there is no such evidence to resample from, resample
        # the first query variable.
        merged = (len(query) == len(self.outputs))
        targets = [o for o in self.outputs if o not in query]
        if merged:
            assert not targets
            targets = [query[0]]
            query = query[1:]
        dataset = self._dataset(targets)
        indices = self.rng.choice(len(dataset), size=samples)
        evidences = [zip(targets, dataset[i]) for i in indices]
        results = [self.simulate(rowid, query, dict(e)) for e in evidences]
        # Make sure to add back the resampled first query variable to results.
        if merged:
            results = [gu.merged(s, e) for s, e in zip(results, evidences)]
        return results

    def logpdf_score(self):
        pass

    def transition(self, N=None):
        return

    # --------------------------------------------------------------------------
    # Internal.

    def _find_neighborhoods(self, query, evidence):
        if not evidence:
            raise ValueError('No evidence in neighbor search: %s.' % evidence)
        if any(np.isnan(v) for v in evidence.values()):
            raise ValueError('Nan evidence in neighbor search: %s.' % evidence)
        # Extract the query, evidence variables from the dataset.
        lookup = list(query) + list(evidence)
        D = self._dataset(lookup)
        # Not enough neighbors: crash for now. Workarounds include:
        # (i) reduce  K, (ii) randomly drop evidences, or (iii) impute dataset.
        if len(D) < self.K:
            raise ValueError('Not enough neighbors: %s.' % ((query, evidence),))
        # Code the dataset with Euclidean embedding.
        D_qr_code = self._dummy_code(D[:,:len(query)], lookup[:len(query)])
        D_ev_code = self._dummy_code(D[:,len(query):], lookup[len(query):])
        D_code = np.column_stack((D_qr_code, D_ev_code))
        # Run nearest neighbor search on the evidence only.
        evidence_code = self._dummy_code([evidence.values()], evidence.keys())
        dist, neighbors = KDTree(D_ev_code).query(evidence_code, k=len(D))
        # Check for equidistant neighbors and possibly extend the search.
        valid = [i for i, d in enumerate(dist[0]) if d <= dist[0][self.K-1]]
        if self.K < len(valid):
            neighbors = self.rng.choice(
                neighbors[0][valid], replace=False, size=self.K)
        else:
            neighbors = neighbors[0][:self.K]
        # For each neighbor, find its nearest M on the full lookup set.
        _, ex = KDTree(D_code).query(D_code[neighbors], k=min(self.M, self.K))
        # Return the dataset and the list of neighborhoods.
        return D[:,:len(query)], ex

    def _create_local_model_joint(self, query, dataset):
        assert all(q in self.outputs for q in query)
        assert dataset.shape[1] == len(query)
        lookup = {
            'numerical': self._create_local_model_numerical,
            'categorical': self._create_local_model_categorical,
            'nominal': self._create_local_model_categorical,
        }
        models = {
            q: lookup[self.stattypes[self.outputs.index(q)]](q, dataset[:,i])
            for i, q in enumerate(query)}
        simulate = lambda q, N=None: {c: models[c].simulate(N) for c in q}
        logpdf = lambda q: sum(models[c].logpdf(x) for c,x in q.iteritems())
        return LocalGpm(simulate, logpdf)

    def _create_local_model_numerical(self, q, locality):
        assert q not in self.levels
        (mu, std) = (np.mean(locality), max(np.std(locality), .01))
        simulate = lambda N=None: self.rng.normal(mu, std, size=N)
        logpdf = lambda x: norm.logpdf(x, mu, std)
        return LocalGpm(simulate, logpdf)

    def _create_local_model_categorical(self, q, locality):
        assert q in self.levels
        assert all(0 <= l < self.levels[q] for l in locality)
        counts = np.bincount(locality.astype(int), minlength=self.levels[q])
        p = counts / np.sum(counts, dtype=float)
        simulate = lambda N: self.rng.choice(self.levels[q], p=p, size=N)
        logpdf = lambda x: np.log(p[x])
        return LocalGpm(simulate, logpdf)

    def _dummy_code(self, D, variables):
        levels = {variables.index(l): self.levels[l]
            for l in variables if l in self.levels}
        return D if not levels\
            else np.asarray([du.dummy_code(r, levels) for r in D])

    def _dataset(self, query):
        indexes = [self.outputs.index(q) for q in query]
        X = np.asarray(self.data.values())[:,indexes]
        return X[~np.any(np.isnan(X), axis=1)]

    def _stattypes(self, query):
        indexes = [self.outputs.index(q) for q in query]
        return [self.stattypes[i] for i in indexes]

    def _populate_evidence(self, rowid, query, evidence):
        if evidence is None:
            evidence = {}
        if rowid in self.data:
            values = self.data[rowid]
            assert len(values) == len(self.outputs)
            evidence_obs = {e:v for e,v in zip(self.outputs, values)
                if not np.isnan(v) and e not in query and e not in evidence
            }
            evidence = gu.merged(evidence, evidence_obs)
        return evidence

    def get_params(self):
        return {}

    def get_distargs(self):
        return {
            'outputs': {
                'stattypes': self.stattypes,
                'statargs': self.statargs,
            },
            'K': self.K,
            'M': self.M,
        }

    @staticmethod
    def name():
        return 'multivariate_knn'

    # --------------------------------------------------------------------------
    # Validation.

    def _validate_init(self, outputs, inputs, K, M, distargs, params, rng):
        # No inputs allowed.
        if inputs:
            raise ValueError('KNN rejects inputs: %s.' % inputs)
        # At least one output.
        if len(outputs) < 2:
            raise ValueError('KNN needs >= 2 outputs: %s.' % outputs)
        # Unique outputs.
        if len(set(outputs)) != len(outputs):
            raise ValueError('Duplicate outputs: %s.' % outputs)
        # Ensure outputs in distargs.
        if not distargs or 'outputs' not in distargs:
            raise ValueError('Missing distargs: %s.' % distargs)
        # Ensure K is positive.
        if K is None or K < 1:
            raise ValueError('Invalid K for nearest neighbors: %s.' % K)
        # Ensure stattypes and statargs in distargs['outputs]'
        if 'stattypes' not in distargs['outputs']\
                or 'statargs' not in distargs['outputs']:
            raise ValueError('Missing output stattypes: %s.' % distargs)
        # Ensure stattypes correct length.
        if len(distargs['outputs']['stattypes']) != len(outputs):
            raise ValueError('Wrong number of stattypes: %s.' % distargs)
        # Ensure statargs correct length.
        if len(distargs['outputs']['statargs']) != len(outputs):
            raise ValueError('Wrong number of statargs: %s.' % distargs)
        # Ensure number of categories provided as k.
        if any('k' not in distargs['outputs']['statargs'][i]
                for i in xrange(len(outputs))
                if distargs['outputs']['stattypes'][i] != 'numerical'):
            raise ValueError('Missing number of categories k: %s' % distargs)

    def _validate_simulate_logpdf(self, rowid, query, evidence, N=None):
        # No invalid number of samples.
        if N is not None and N <= 0:
            raise ValueError('Unknown number of samples: %s.' % N)
        # At least K observations before we can do K nearest neighbors.
        if self.N < self.K:
            raise ValueError(
                'MultivariateKnn needs at least %d observations: %d.'
                % (self.K, self.N))
        # Need a query set.,
        if not query:
            raise ValueError('No query specified: %s.' % query)
        # No query variables not in outputs.
        if any(q not in self.outputs for q in query):
            raise ValueError(
                'Unknown output variables in query: (%s,%s).'
                % (self.outputs, query))
        # No evidence variables not in outputs.
        if any(e not in self.outputs for e in evidence):
            raise ValueError(
                'Unknown output variables in evidence: (%s,%s).'
                % (self.outputs, evidence))
        # No duplicate variables in query and evidence.
        if any(q in evidence for q in query):
            raise ValueError(
                'Duplicate variable in query and evidence: (%s,%s).'
                % (query, evidence))
        # Check for a nan in evidence.
        if any(np.isnan(v) for v in evidence.itervalues()):
            raise ValueError('Nan value in evidence: %s.' % evidence)
        # Check for a nan in query.,
        if isinstance(query, dict)\
                and any(np.isnan(v) for v in query.itervalues()):
            raise ValueError('Nan value in query: %s.' % query)

    def _validate_incorporate(self, rowid, query, evidence):
        # No duplicate observation.
        if rowid in self.data:
            raise ValueError('Already observed: %d.' % rowid)
        # No evidence.
        if evidence:
            raise ValueError('No evidence allowed: %s.' % evidence)
        # Missing query.
        if not query:
            raise ValueError('No query specified: %s.' % query)
        # No unknown variables.
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))

    def _validate_unincorporate(self, rowid):
        if rowid not in self.data:
            raise ValueError('No such observation: %d.' % rowid)

    # --------------------------------------------------------------------------
    # Serialization.

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['distargs'] = self.get_distargs()
        metadata['N'] = self.N
        metadata['data'] = self.data.items()

        metadata['params'] = dict()

        metadata['factory'] = ('cgpm.knn.mvknn', 'MultivariateKnn')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        knn = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            K=metadata['distargs']['K'],    # Pending migration to **kwargs
            M=metadata['distargs']['M'],
            distargs=metadata['distargs'],
            params=metadata['params'],
            rng=rng)
        knn.data = OrderedDict(metadata['data'])
        knn.N = metadata['N']
        return knn
