# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import base64
import copy
import math

from collections import defaultdict
from datetime import datetime

import venture.shortcuts as vs

from cgpm.cgpm import CGpm
from cgpm.utils import config as cu
from cgpm.utils import general as gu


class VsCGpm(CGpm):
    """CGpm specified by a Venturescript program."""

    def __init__(self, outputs, inputs, rng=None, sp=None, **kwargs):
        # Set the rng.
        self.rng = rng if rng is not None else gu.gen_rng(1)
        seed = self.rng.randint(1, 2**31 - 1)
        # Basic input and output checking.
        if len(set(outputs)) != len(outputs):
            raise ValueError('Non unique outputs: %s' % outputs)
        if len(set(inputs)) != len(inputs):
            raise ValueError('Non unique inputs: %s' % inputs)
        if not all(o not in inputs for o in outputs):
            raise ValueError('Duplicates: %s, %s' % (inputs, outputs))
        if not all(i not in outputs for i in inputs):
            raise ValueError('Duplicates: %s, %s' % (inputs, outputs))
        # Retrieve the ripl.
        self.ripl = kwargs.get('ripl', vs.make_lite_ripl(seed=seed))
        self.mode = kwargs.get('mode', 'church_prime')
        # Execute the program.
        self.source = kwargs.get('source', None)
        if self.source is not None:
            self.ripl.set_mode(self.mode)
            self.ripl.execute_program(self.source)
        # Load any plugins.
        self.plugins = kwargs.get('plugins', None)
        if self.plugins:
            for plugin in self.plugins.split(','):
                self.ripl.load_plugin(plugin.strip())
        # Force the mode to church_prime.
        self.ripl.set_mode('church_prime')
        # Create the CGpm.
        if not kwargs.get('supress', None):
            self.ripl.evaluate('(%s)' % ('make_cgpm' if sp is None else sp,))
        # Check correct outputs.
        if len(outputs) != len(self.ripl.sample('simulators')):
            raise ValueError('source.simulators list disagrees with outputs.')
        self.outputs = outputs
        # Check correct inputs.
        if len(inputs) != self.ripl.evaluate('(size inputs)'):
            raise ValueError('source.inputs list disagrees with inputs.')
        self.inputs = inputs
        # Check overriden observers.
        if len(self.outputs) != self.ripl.evaluate('(size observers)'):
            raise ValueError('source.observers list disagrees with outputs.')
        # XXX Eliminate this nested defaultdict
        # Inputs and labels for incorporate/unincorporate.
        self.obs = defaultdict(lambda: defaultdict(dict))

    def incorporate(self, rowid, observation, inputs=None):
        inputs = self._validate_incorporate(rowid, observation, inputs)
        for variable, value in observation.iteritems():
            self._observe_cell(rowid, variable, value, inputs)

    def unincorporate(self, rowid):
        if rowid not in self.obs:
            raise ValueError('Never incorporated: %d' % rowid)
        for q in self.outputs:
            self._forget_cell(rowid, q)
        assert len(self.obs[rowid]['labels']) == 0
        del self.obs[rowid]

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        return 0

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        constraints_clean, inputs_clean = \
            self._validate_simulate(rowid, targets, constraints, inputs)
        # Handle constraints on the multivariate output.
        if constraints_clean:
            # Observe constrained outputs.
            for variable, value in constraints_clean.iteritems():
                self._observe_cell(rowid, variable, value, inputs_clean)
            # Run local inference in rowid scope, with 15 steps of MH.
            self.ripl.infer('(resimulation_mh (atom %i) all %i)' % (rowid, 15))
        # Retrieve samples, with 5 steps of MH between predict.
        def retrieve_sample(q, l):
            # XXX Only run inference on the latent variables in the block.
            # self.ripl.infer('(resimulation_mh (atom %i) all %i)' % (rowid, 5))
            return self._predict_cell(rowid, q, inputs_clean, l)
        labels = [self._gen_label() for q in targets]
        samples = {q: retrieve_sample(q, l) for q, l in zip(targets, labels)}
        # Forget predicted targets variables.
        for label in labels:
            self.ripl.forget(label)
        # Forget constrained outputs.
        for q, _v in constraints_clean.iteritems():
            self._forget_cell(rowid, q)
        return samples

    def logpdf_score(self):
        raise NotImplementedError

    def transition(self, program=None, N=None):
        if program is None:
            num_iters = N if N is not None else 1
            self.ripl.infer('(transition %d)' % (num_iters,))
        else:
            self.ripl.execute_program(program)

    def to_metadata(self):
        metadata = dict()
        metadata['source'] = self.source
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['mode'] = self.mode
        metadata['plugins'] = self.plugins
        # Save the observations. We need to convert integer keys to strings.
        metadata['obs'] = VsCGpm._obs_to_json(copy.deepcopy(self.obs))
        metadata['binary'] =  base64.b64encode(self.ripl.saves())
        metadata['factory'] = ('cgpm.venturescript.vscgpm', 'VsCGpm')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        ripl = vs.make_lite_ripl()
        ripl.loads(base64.b64decode(metadata['binary']))
        cgpm = VsCGpm(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            ripl=ripl,
            source=metadata['source'],
            mode=metadata['mode'],
            supress=True,
            plugins=metadata['plugins'],
            rng=rng,
        )
        # Restore the observations. We need to convert string keys to integers.
        # XXX Eliminate this terrible defaultdict hack. See Github #187.
        obs_converted = VsCGpm._obs_from_json(metadata['obs'])
        cgpm.obs = defaultdict(lambda: defaultdict(dict))
        for key, value in obs_converted.iteritems():
            cgpm.obs[key] = defaultdict(dict, value)
        return cgpm

    # --------------------------------------------------------------------------
    # Internal helpers.

    def _predict_cell(self, rowid, target, inputs, label):
        inputs_list = [inputs[i] for i in self.inputs]
        sp_args = str.join(' ', map(str, [rowid] + inputs_list))
        i = self.outputs.index(target)
        return self.ripl.predict(
            '((lookup simulators %i) %s)' % (i, sp_args), label=label)

    def _observe_cell(self, rowid, query, value, inputs):
        inputs_list = [inputs[i] for i in self.inputs]
        label = '\''+self._gen_label()
        sp_args = str.join(' ', map(str, [rowid] + inputs_list + [value, label]))
        i = self.outputs.index(query)
        self.ripl.evaluate('((lookup observers %i) %s)' % (i, sp_args))
        self.obs[rowid]['labels'][query] = label[1:]

    def _forget_cell(self, rowid, query):
        if query not in self.obs[rowid]['labels']:
            return
        label = self.obs[rowid]['labels'][query]
        self.ripl.forget(label)
        del self.obs[rowid]['labels'][query]

    def _gen_label(self):
        return 't%s%s' % (
            self.rng.randint(1,100),
            datetime.now().strftime('%Y%m%d%H%M%S%f'))

    def _validate_incorporate(self, rowid, observation, inputs=None):
        inputs = inputs or {}
        if not observation:
            raise ValueError('No observation: %s.' % observation)
        # All inputs present, and no nan values.
        if rowid not in self.obs and set(inputs) != set(self.inputs):
            raise ValueError('Missing inputs: %s, %s'
                % (inputs, self.inputs))
        if not set.issubset(set(observation), set(self.outputs)):
            raise ValueError('Unknown observation: %s,%s'
                % (observation, self.outputs))
        if any(math.isnan(inputs[i]) for i in inputs):
            raise ValueError('Nan inputs: %s' % inputs)
        if any(math.isnan(observation[i]) for i in observation):
            raise ValueError('Nan observation: %s' % (observation,))
        # Inputs optional if (rowid,q) previously incorporated, or must match.
        if rowid in self.obs:
            if any(q in self.obs[rowid]['labels'] for q in observation):
                raise ValueError('Observation exists: %d, %s'
                    % (rowid, observation))
            if inputs:
                self._check_matched_inputs(rowid, inputs)
        else:
            self.obs[rowid]['inputs'] = dict(inputs)
        return self.obs[rowid]['inputs']

    def _validate_simulate(self, rowid, targets, constraints, inputs):
        constraints = constraints or dict()
        inputs = inputs or dict()
        if rowid not in self.obs and set(inputs) != set(self.inputs):
            raise ValueError('Missing inputs: %s, %s' % (inputs, self.inputs))
        if any(math.isnan(inputs[i]) for i in inputs):
            raise ValueError('Nan inputs: %s' % (inputs,))
        if any(math.isnan(constraints[i]) for i in constraints):
            raise ValueError('Nan constraints: %s' % (constraints,))
        if not all(i in self.inputs for i in inputs):
            raise ValueError('Unknown inputs: %s' % (inputs,))
        if not all(i in self.outputs for i in constraints):
            raise ValueError('Unknown constraints: %s' % (constraints,))
        if not all(i in self.outputs for i in targets):
            raise ValueError('Unknown targets: %s' % (targets,))
        if set.intersection(set(targets), set(constraints)):
            raise ValueError('Overlapping targets and constraints: %s, %s'
                % (targets, constraints,))
        if rowid in self.obs:
            if any(q in self.obs[rowid]['labels'] for q in constraints):
                raise ValueError('Constrained observation exists: %d, %s, %s'
                    % (rowid, targets, constraints))
            if inputs:
                self._check_matched_inputs(rowid, inputs)
            else:
                inputs = self.obs[rowid]['inputs']
        assert set(inputs) == set(self.inputs)
        return constraints, inputs

    def _check_matched_inputs(self, rowid, inputs):
        # Avoid creating 'inputs' by the defaultdict.
        exists = (rowid in self.obs) and ('inputs' in self.obs[rowid])
        if exists and inputs != self.obs[rowid]['inputs']:
            raise ValueError('Given inputs contradicts dataset: %d, %s, %s' %
                (rowid, inputs, self.obs[rowid]['inputs']))

    @staticmethod
    def _obs_to_json(obs):
        def convert_key_int_to_str(d):
            assert all(isinstance(c, int) for c in d)
            return {str(c): v for c, v in d.iteritems()}
        obs2 = convert_key_int_to_str(obs)
        for r in obs2:
            obs2[r]['inputs'] = convert_key_int_to_str(obs2[r]['inputs'])
            obs2[r]['labels'] = convert_key_int_to_str(obs2[r]['labels'])
        return obs2

    @staticmethod
    def _obs_from_json(obs):
        def convert_key_str_to_int(d):
            assert all(isinstance(c, (str, unicode)) for c in d)
            return {int(c): v for c, v in d.iteritems()}
        obs2 = convert_key_str_to_int(obs)
        for r in obs2:
            obs2[r]['inputs'] = convert_key_str_to_int(obs2[r]['inputs'])
            obs2[r]['labels'] = convert_key_str_to_int(obs2[r]['labels'])
        return obs2
