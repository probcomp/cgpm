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

from statsmodels.nonparameteric import kernel_density

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class MultivariateKde(CGpm):
    """Multivariate Kernel Density Estimation support continuous and categorical
    datatypes.

    [1] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)

    This implementation extends the baseline implementation of [1] from the
    statsmodels package to satisfy the CGPM interface. In particular, it is
    extended to support (un)conditional simulation by importance weighting the
    dataset.
    """

    def __init__(self, outputs, inputs, distargs=None, params=None,
            rng=None):
        # Default parameter settings.
        if params is None:
            params = {}
        # Entropy.
        if rng is None:
            rng = gu.gen_rng(1)
        # No inputs.
        if inputs:
            raise ValueError('KDE rejects inputs: %s.' % inputs)
        # Correct outputs.
        if len(outputs) < 1:
            raise ValueError('KDE needs >= 1 outputs: %s.' % outputs)
        if len(set(outputs)) != len(outputs):
            raise ValueError('Duplicate outputs: %s.' % outputs)
        # Statistical types.
        if not distargs or 'outputs' not in distargs:
            raise ValueError('Missing distargs: %s.' % distargs)
        if 'stattypes' not in distargs['outputs']:
            raise ValueError('Missing statistical types: %s.' % distargs)
        # Build the object.
        self.rng = rng
        # Varible indexes.
        self.outputs = outputs
        self.inputs = []
        # Distargs.
        self.distargs = distargs['outputs']['stattypes']
        # Dataset.
        self.data = OrderedDict()
        self.N = 0
        # Parameters of the kernels.
        self.bw = params.get('bw', [1.]*len(self.outputs))

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
        # Incorporate observed variables.
        x = [query.get(q, np.nan) for q in self.outputs]
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
        if any(np.isnan(v) for v in query.values()):
            raise ValueError('Cannot query nan values: %s' % query)
        if any(q not in self.outputs for q in query):
            raise ValueError('Unknown variables: (%s,%s).'
                % (query, self.outputs))
        if any(q in evidence for q in query):
            raise ValueError('Duplicate variable: (%s,%s).' % (query, evidence))
        if not evidence:
            model = kernel_density.KDEMultivariate(
                self._dataset(query),
                self._stattypes(query),
                bw=self._bw(query))
            pdf = model.pdf(query.values())
        else:
            model = kernel_density.KDEMultivariateConditional(
                self._dataset(query),
                self._dataset(evidence),
                self._stattypes(query),
                self._stattypes(evidence),
                bw=self._bw(query) + self._bw(evidence))
            pdf = model.pdf(query.values(), evidence.values())
        return np.log(pdf)

    def simulate(self, rowid, query, evidence=None, N=None):
        pass

    def logpdf_score(self):
        def compute_logpdf(x):
            assert len(x) == self.D
            query = {i:v for i,v in enumerate(x) if not np.isnan(v)}
            return self.logpdf(-1, query, evidence=None)
        return sum(compute_logpdf(x) for x in self.data)

    def transition(self, N=None):
        kde = kernel_density.KDEMultivariate(
            self._dataset(self.outputs),
            self._stattypes(self.outputs),
            bw='cv_ml')
        self.bw = kde.bw.tolist()

    # --------------------------------------------------------------------------
    # Internal.

    def _dataset(self, query):
        indexes = [self.outputs.index(q) for q in query]
        X = np.asarray(self.data.values())[:,indexes]
        return X[~np.any(np.isnan(X))]

    def _bw(self, query):
        indexes = [self.outputs.index(q) for q in query]
        return [self.bw[i] for i in indexes]

    def _stattypes(self, query):
        indexes = [self.outputs.index(q) for q in query]
        lookup = {
            'c': 'numerical',
            'u': 'nominal',
        }
        return str.join(',', [lookup[self.stattypes[q]] for q in query])

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

    def get_params(self):
        return {
            'bw': self.bw,
        }

    def get_distargs(self):
        return {
            'outputs': {
                'stattypes': self.stattypes,
            },
        }

    @staticmethod
    def name():
        return 'multivariate_kde'

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
        metadata['params']['bw'] = self.bw

        metadata['factory'] = ('cgpm.kde.mvkde', 'MultivariateKde')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        kde = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            distargs=metadata['distargs'],
            params=metadata['params'],
            rng=rng)
        kde.data = OrderedDict(metadata['data'])
        kde.N = metadata['N']
        return kde
