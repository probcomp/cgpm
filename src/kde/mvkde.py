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

from statsmodels.nonparametric import _kernel_base
from statsmodels.nonparametric import kernel_density

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class MultivariateKde(CGpm):
    """Multivariate Kernel Density Estimation support continuous and categorical
    datatypes.

    [1] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)

    This implementation extends the baseline implementation of [1] from the
    statsmodels package to satisfy the CGPM interface. In particular, it is
    extended to support conditional simulation by importance weighting the
    exemplars.
    """

    def __init__(self, outputs, inputs, distargs=None, params=None,
            rng=None):
        # Default arguments.
        if params is None:
            params = {}
        if rng is None:
            rng = gu.gen_rng(1)
        # No inputs allowed.
        if inputs:
            raise ValueError('KDE rejects inputs: %s.' % inputs)
        # At least one output.
        if len(outputs) < 1:
            raise ValueError('KDE needs >= 1 outputs: %s.' % outputs)
        # Unique outputs.
        if len(set(outputs)) != len(outputs):
            raise ValueError('Duplicate outputs: %s.' % outputs)
        # Ensure outputs in distargs.
        if not distargs or 'outputs' not in distargs:
            raise ValueError('Missing distargs: %s.' % distargs)
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
        # Parameters of the kernels.
        self.bw = params.get('bw', [self._default_bw(o) for o in self.outputs])

    def incorporate(self, rowid, observation, inputs=None):
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
        # Incorporate observed variables.
        x = [observation.get(q, np.nan) for q in self.outputs]
        # Update dataset and counts.
        self.data[rowid] = x
        self.N += 1

    def unincorporate(self, rowid):
        try:
            del self.data[rowid]
        except KeyError:
            raise ValueError('No such observation: %d.' % rowid)
        self.N -= 1

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        if self.N == 0:
            raise ValueError('KDE requires at least one observation.')
        constraints = self.populate_constraints(rowid, targets, constraints)
        if inputs:
            raise ValueError('Prohibited inputs: %s' % (inputs,))
        if not targets:
            raise ValueError('No targets: %s' % (targets,))
        if any(np.isnan(v) for v in targets.values()):
            raise ValueError('Invalid nan values in targets: %s' % (targets,))
        if any(q not in self.outputs for q in targets):
            raise ValueError('Unknown targets: %s' % (targets,))
        if any(q in constraints for q in targets):
            raise ValueError('Duplicate variable: %s, %s'
                % (targets, constraints,))
        if not constraints:
            model = kernel_density.KDEMultivariate(
                self._dataset(targets),
                self._stattypes(targets),
                bw=self._bw(targets),
            )
            pdf = model.pdf(targets.values())
        else:
            full_members = self._dataset(targets.keys() + constraints.keys())
            model = kernel_density.KDEMultivariateConditional(
                full_members[:,:len(targets)],
                full_members[:,len(targets):],
                self._stattypes(targets),
                self._stattypes(constraints),
                bw=np.concatenate((self._bw(targets), self._bw(constraints))),
            )
            pdf = model.pdf(targets.values(), constraints.values())
        return np.log(pdf)

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        if self.N == 0:
            raise ValueError('KDE requires at least one observation.')
        if inputs:
            raise ValueError('Prohibited inputs: %s' % (inputs,))
        if not targets:
            raise ValueError('No targets: %s' % (targets,))
        if any(q not in self.outputs for q in targets):
            raise ValueError('Unknown targets: %s' % (targets,))
        if constraints and any(q in constraints for q in targets):
            raise ValueError('Duplicate variable: %s, %s'
                % (targets, constraints,))
        constraints = self.populate_constraints(rowid, targets, constraints)
        if constraints:
            full_members = self._dataset(targets + constraints.keys())
            weights = _kernel_base.gpke(
                self._bw(constraints),
                full_members[:,len(targets):],
                constraints.values(),
                self._stattypes(constraints),
                tosum=False,
            )
            targets_members = full_members[:,:len(targets)]
        else:
            targets_members = self._dataset(targets)
            weights = [1] * len(targets_members)
        assert len(weights) == len(targets_members)
        index = gu.pflip(weights, size=N, rng=self.rng)
        if N is None:
            return self._simulate_member(targets_members[index], targets)
        return [self._simulate_member(targets_members[i], targets) for i in index]

    def _simulate_member(self, row, targets):
        assert len(row) == len(targets)
        lookup = {
            'c': self._simulate_gaussian_kernel,
            'u': self._simulate_aitchison_aitken_kernel
        }
        funcs = [lookup[s] for s in self._stattypes(targets)]
        return {q: f(q, v) for f, q, v in zip(funcs, targets, row)}

    def _simulate_gaussian_kernel(self, q, Xi):
        idx = self.outputs.index(q)
        assert self.stattypes[idx] == 'numerical'
        return self.rng.normal(loc=Xi, scale=self.bw[idx])

    def _simulate_aitchison_aitken_kernel(self, q, Xi):
        idx = self.outputs.index(q)
        assert self.stattypes[idx] in ['categorical', 'nominal']
        c = self.levels[q]
        def _compute_probabilities(s):
            return 1 - self.bw[idx] if s == Xi else self.bw[idx] / (c - 1)
        probs = map(_compute_probabilities, range(c))
        assert np.allclose(sum(probs), 1)
        return self.rng.choice(range(c), p=probs)

    def logpdf_score(self):
        def compute_logpdf(rowid, x):
            assert len(x) == len(self.outputs)
            targets = {self.outputs[i]: v
                for i, v in enumerate(x) if not np.isnan(v)}
            return self.logpdf(rowid, targets)
        return sum(compute_logpdf(rowid, x) for rowid, x in self.data.items())

    def transition(self, N=None):
        if self.N > 0:
            dataset = self._dataset(self.outputs)
            stattypes = self._stattypes(self.outputs)
            # Learn the kernel bandwidths.
            kde = kernel_density.KDEMultivariate(
                dataset, stattypes, bw='cv_ml')
            self.bw = kde.bw.tolist()

    # --------------------------------------------------------------------------
    # Internal.

    def _dataset(self, outputs):
        indexes = [self.outputs.index(q) for q in outputs]
        X = np.asarray(self.data.values())[:,indexes]
        return X[~np.any(np.isnan(X), axis=1)]

    def _default_bw(self, q):
        i = self.outputs.index(q)
        return 1 if self.stattypes[i] == 'numerical' else .1

    def _bw(self, outputs):
        indexes = [self.outputs.index(q) for q in outputs]
        return np.asarray([self.bw[i] for i in indexes])

    def _stattypes(self, outputs):
        indexes = [self.outputs.index(q) for q in outputs]
        lookup = {
            'numerical': 'c',
            'categorical': 'u',
            'nominal': 'u',
        }
        return str.join('', [lookup[self.stattypes[i]] for i in indexes])

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
        return {
            'bw': self.bw,
        }

    def get_distargs(self):
        return {
            'outputs': {
                'stattypes': self.stattypes,
                'statargs': self.statargs,
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
