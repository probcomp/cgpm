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
    """Multivariate K-Nearest-Neighbors builds local statistical models on a
    per-query basis.

    Algorithm for simulate(rowid, targets, constraints) and
    logpdf(rowid, targets, constraints):

        - Find K nearest neighbors to `rowid` based only on the `constraints`.

        - For each nearest neighbor k = 1,...,K

            - Find M nearest neighbors of k (called the locality of k) based
            on both the `constraints` and `targets` dimensions.

            - For each target variable q \in target:

                - Learn a primitive univariate CGPM,  using the dimension q of
                the M neighbors in the locality of k.

            - Return a product CGPM G_k representing locality k.

        Overall CGPM G = (1/K)*G_1 + ... + (1/K)*G_K is a simple-weighted
        mixture of the product CGPM learned in each locality.

    This "locality-based" algorithm is designed to capture the dependence
    between the target variables, rather than assume that all the target
    variables are independent conditioned on the constraints. Github ticket #133
    will support selecting either the independent or locality-based versions of
    KNN.
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

    def incorporate(self, rowid, observation, inputs=None):
        self._validate_incorporate(rowid, observation, inputs)
        # Incorporate observed variables.
        x = [observation.get(q, np.nan) for q in self.outputs]
        # Update dataset and counts.
        self.data[rowid] = x
        self.N += 1

    def unincorporate(self, rowid):
        self._validate_unincorporate(rowid)
        del self.data[rowid]
        self.N -= 1

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        constraints = self.populate_constraints(rowid, targets, constraints)
        # XXX Disable logpdf queries without constraints.
        if inputs:
            raise ValueError('Prohibited inputs: %s' % (inputs,))
        if not constraints:
            raise ValueError('Provide at least one constraint: %s'
                % (constraints,))
        self._validate_simulate_logpdf(rowid, targets, constraints)
        # Retrieve the dataset and neighborhoods.
        dataset, neighborhoods = self._find_neighborhoods(targets, constraints)
        models = [self._create_local_model_joint(targets, dataset[n])
            for n in neighborhoods]
        # Compute logpdf in each neighborhood and simple average.
        lp = [m.logpdf(targets) for m in models]
        return gu.logsumexp(lp) - np.log(len(models))

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        if inputs:
            raise ValueError('Prohibited inputs: %s' % (inputs,))
        N_sim = 1 if N is None else N
        constraints = self.populate_constraints(rowid, targets, constraints)
        self._validate_simulate_logpdf(rowid, targets, constraints, N_sim)
        if constraints:
            # Retrieve the dataset and neighborhoods.
            dataset, neighborhoods = self._find_neighborhoods(
                targets, constraints)
            models = [self._create_local_model_joint(targets, dataset[n])
                for n in neighborhoods]
            # Sample the models.
            indices = self.rng.choice(len(models), size=N_sim)
            # Sample from each model.
            sampled_models = [models[i] for i in indices]
            results = [m.simulate(targets) for m in sampled_models]
        else:
            results = self._simulate_fallback(rowid, targets, N_sim)
            assert len(results) == N_sim
        return results[0] if N is None else results

    def _simulate_fallback(self, rowid, targets, N):
        # Fallback: if there is no such constraints to resample from, then
        # resample the first variable.
        merged = len(targets) == len(self.outputs)
        targets_dummy = [o for o in self.outputs if o not in targets]
        if merged:
            assert not targets_dummy
            targets_dummy = [targets[0]]
            targets = targets[1:]
        dataset = self._dataset(targets_dummy)
        indices = self.rng.choice(len(dataset), size=N)
        constraints = [zip(targets_dummy, dataset[i]) for i in indices]
        results = [self.simulate(rowid, targets, dict(e)) for e in constraints]
        # Make sure to add back the resampled first target variable to results.
        if merged:
            results = [gu.merged(s, e) for s, e in zip(results, constraints)]
        return results

    def logpdf_score(self):
        pass

    def transition(self, N=None):
        return

    # --------------------------------------------------------------------------
    # Internal.

    def _find_neighborhoods(self, targets, constraints):
        if not constraints:
            raise ValueError('No constraints in neighbor search.')
        if any(np.isnan(v) for v in constraints.values()):
            raise ValueError('Nan constraints in neighbor search.')
        # Extract the targets, constraints from the dataset.
        lookup = list(targets) + list(constraints)
        D = self._dataset(lookup)
        # Not enough neighbors: crash for now. Workarounds include:
        # (i) reduce K, (ii) randomly drop constraints, (iii) impute dataset.
        if len(D) < self.K:
            raise ValueError('Not enough neighbors: %s'
                % ((targets, constraints),))
        # Code the dataset with Euclidean embedding.
        N = len(targets)
        D_qr_code = self._dummy_code(D[:,:N], lookup[:N])
        D_ev_code = self._dummy_code(D[:,N:], lookup[N:])
        D_code = np.column_stack((D_qr_code, D_ev_code))
        # Run nearest neighbor search on the constraints only.
        constraints_code = self._dummy_code(
            [constraints.values()], constraints.keys())
        dist, neighbors = KDTree(D_ev_code).query(constraints_code, k=len(D))
        # Check for equidistant neighbors and possibly extend the search.
        valid = [i for i, d in enumerate(dist[0]) if d <= dist[0][self.K-1]]
        if self.K < len(valid):
            neighbors = self.rng.choice(neighbors[0][valid],
                replace=False, size=self.K)
        else:
            neighbors = neighbors[0][:self.K]
        # For each neighbor, find its nearest M on the full lookup set.
        _, ex = KDTree(D_code).query(D_code[neighbors], k=min(self.M, self.K))
        # Return the dataset and the list of neighborhoods.
        return D[:,:len(targets)], ex

    def _create_local_model_joint(self, targets, dataset):
        assert all(q in self.outputs for q in targets)
        assert dataset.shape[1] == len(targets)
        lookup = {
            'numerical': self._create_local_model_numerical,
            'categorical': self._create_local_model_categorical,
            'nominal': self._create_local_model_categorical,
        }
        models = {
            q: lookup[self.stattypes[self.outputs.index(q)]](q, dataset[:,i])
            for i, q in enumerate(targets)}
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

    def _dataset(self, outputs):
        indexes = [self.outputs.index(q) for q in outputs]
        X = np.asarray(self.data.values())[:,indexes]
        return X[~np.any(np.isnan(X), axis=1)]

    def _stattypes(self, outputs):
        indexes = [self.outputs.index(q) for q in outputs]
        return [self.stattypes[i] for i in indexes]

    def populate_constraints(self, rowid, targets, constraints):
        if constraints is None:
            constraints = {}
        if rowid in self.data:
            values = self.data[rowid]
            assert len(values) == len(self.outputs)
            observations = {
                output : value
                for output, value in zip(self.outputs, values)
                if not np.isnan(value)
                    and output not in targets
                    and output not in constraints
            }
            constraints = gu.merged(constraints, observations)
        return constraints

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

    def _validate_simulate_logpdf(self, rowid, targets, constraints, N=None):
        # No invalid number of samples.
        if N is not None and N <= 0:
            raise ValueError('Unknown number of samples: %s.' % N)
        # At least K observations before we can do K nearest neighbors.
        if self.N < self.K:
            raise ValueError('MultivariateKnn needs >= %d observations: %d'
                % (self.K, self.N))
        # Need targets.
        if not targets:
            raise ValueError('No targets specified: %s.' % targets)
        # All targets in outputs.
        if any(q not in self.outputs for q in targets):
            raise ValueError('Unknown variables in targets: %s, %s'
                % (self.outputs, targets))
        # All constraints in outputs.
        if any(e not in self.outputs for e in constraints):
            raise ValueError('Unknown variables in constraints: %s,%s'
                % (self.outputs, constraints))
        # No duplicate variables in targets and constraints.
        if any(q in constraints for q in targets):
            raise ValueError('Duplicate variable in targets/constraints: %s %s'
                % (targets, constraints))
        # Check for a nan in constraints.
        if any(np.isnan(v) for v in constraints.itervalues()):
            raise ValueError('Nan value in constraints: %s.' % constraints)
        # Check for a nan in targets.,
        if isinstance(targets, dict)\
                and any(np.isnan(v) for v in targets.itervalues()):
            raise ValueError('Nan value in targets: %s.' % targets)

    def _validate_incorporate(self, rowid, observation, inputs):
        # No duplicate observation.
        if rowid in self.data:
            raise ValueError('Already observed: %d.' % rowid)
        # No inputs.
        if inputs:
            raise ValueError('No inputs allowed: %s.' % inputs)
        # Missing observation.
        if not observation:
            raise ValueError('No observation specified: %s.' % observation)
        # No unknown variables.
        if any(q not in self.outputs for q in observation):
            raise ValueError('Unknown variables: (%s,%s).'
                % (observation, self.outputs))

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
