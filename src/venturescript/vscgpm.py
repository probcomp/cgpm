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
import math
import os

from datetime import datetime

import venture.shortcuts as vs

from venture.exception import VentureException

from cgpm.cgpm import CGpm
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
        self.ripl = VsCGpm._load_helpers(self.ripl)
        # Execute the program.
        self.source = kwargs.get('source', None)
        if self.source is not None:
            self.ripl.set_mode(self.mode)
            self.ripl.execute_program(self.source)
        # Load any additional plugins.
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
        if len(outputs) != len(self.ripl.sample('outputs')):
            raise ValueError('source.outputs list disagrees with outputs.')
        self.outputs = outputs
        # Check correct inputs.
        if len(inputs) != self.ripl.sample('(size inputs)'):
            raise ValueError('source.inputs list disagrees with inputs.')
        self.inputs = inputs
        self.input_mapping = self._get_input_mapping(self.inputs)
        # Check custom observers.
        num_observers = self._get_num_observers()
        self.observe_custom = num_observers is not None
        if self.observe_custom and len(self.outputs) != num_observers:
            raise ValueError('source.observers list disagrees with outputs.')
        # Entry labels[rowid][query] is label used to observe output cell.
        self.labels = dict()

    def incorporate(self, rowid, observation, inputs=None):
        observation_clean = self._cleanse_observation(rowid, observation)
        inputs_clean = self._cleanse_inputs(rowid, inputs)
        for cin, value in inputs_clean.iteritems():
            self._observe_input_cell(rowid, cin, value)
        for cout, value in observation_clean.iteritems():
            self._observe_output_cell(rowid, cout, value)

    def unincorporate(self, rowid):
        if rowid not in self.labels:
            raise ValueError('Never incorporated: %d' % rowid)
        for cout in self.outputs:
            if self._is_observed_output_cell(rowid, cout):
                self._forget_output_cell(rowid, cout)
        for cin in self.inputs:
            if self._is_observed_input_cell(rowid, cin):
                self._forget_input_cell(rowid, cin)
        assert rowid not in self.labels

    def logpdf(self, rowid, targets, constraints=None, inputs=None,
            num_samples=10, num_steps=10):
        inputs_clean = self._cleanse_inputs(rowid, inputs)
        targets_clean = self._cleanse_targets(rowid, targets)
        constraints_clean = self._cleanse_constraints(
            rowid, targets, constraints)
        # Observe any unseen inputs.
        for cin, value in inputs_clean.iteritems():
            self._observe_input_cell(rowid, cin, value)
        # Observe any unobserved constrained outputs.
        for cout, value in constraints_clean.iteritems():
            self._observe_output_cell(rowid, cout, value)
        # Observe the targets.
        for cout, value in targets_clean.iteritems():
            self._observe_output_cell(rowid, cout, value)
        # Compute probabilities of each target given its parents.]
        logps = [None] * num_steps
        for step in xrange(num_steps):
            self.ripl.infer('(mh (atom %i) all %i)' % (rowid, num_steps))
            logps[step] = sum([self._logpdf_output_cell(rowid, cout)
                for cout in targets_clean])
        # Forget observed targets.
        for cout in targets_clean:
            self._forget_output_cell(rowid, cout)
        # Forget observed constraints.
        for cout in constraints_clean:
            self._forget_output_cell(rowid, cout)
        # Forget observed inputs.
        for cin in inputs_clean:
            self._forget_input_cell(rowid, cin)
        return gu.logmeanexp(logps)

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        inputs_clean = self._cleanse_inputs(rowid, inputs)
        targets_clean = self._cleanse_targets(rowid, targets)
        constraints_clean = self._cleanse_constraints(
            rowid, targets, constraints)
        # Observe any unseen inputs.
        for cin, value in inputs_clean.iteritems():
            self._observe_input_cell(rowid, cin, value)
        # Observe any unobserved constrained outputs.
        for cout, value in constraints_clean.iteritems():
            self._observe_output_cell(rowid, cout, value)
        # Run local inference in rowid scope, with 15 steps of MH.
        self.ripl.infer('(mh (atom %i) all %i)' % (rowid, 15))
        # Generate labels and predictions of outputs.
        labels = [self._gen_label() for _cout in targets]
        samples = {cout: self._predict_output_cell(rowid, cout, label)
            for cout, label in zip(targets_clean, labels)}
        # Forget predicted targets.
        for label in labels:
            self.ripl.forget(label)
        # Forget observed constraints.
        for cout in constraints_clean:
            self._forget_output_cell(rowid, cout)
        # Forget observed inputs.
        for cin in inputs_clean:
            self._forget_input_cell(rowid, cin)
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
        metadata['labels'] = VsCGpm.convert_key_int_to_str(self.labels)
        metadata['binary'] =  base64.b64encode(self.ripl.saves())
        metadata['factory'] = ('cgpm.venturescript.vscgpm', 'VsCGpm')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        ripl = VsCGpm._load_helpers(vs.make_lite_ripl())
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
        labels = VsCGpm.convert_key_str_to_int(metadata['labels'])
        for rowid, mapping in labels.iteritems():
            cgpm.labels[rowid] = mapping
        return cgpm

    # --------------------------------------------------------------------------
    # Internal helpers.

    def _logpdf_output_cell(self, rowid, cout):
        # Assess density of node x (scope:rowid, block:cout) given parents.
        # Let node y->x->z represent link structure in the dependency graph:
        #   logp_joint          = p(x,z|y)
        #   logp_likelihood     = p(z|x,y)
        #   logp_at             = logp_joint - logp_likelihood
        #                           = log(p(x,z|y)/p(z|x,y))
        #                           = log(p(z|x,y)p(x|y)/p(z|x,y))
        #                           = log(p(x|y))
        output_idx = self.outputs.index(cout)
        sp_rowid = '(atom %d)' % (rowid,)
        lp_joint = self.ripl.evaluate('(log_joint_at %s %s)'
            % (sp_rowid, output_idx))
        logp_likelihood = self.ripl.evaluate('(log_likelihood_at %s %s)'
            % (sp_rowid, output_idx))
        return lp_joint[0] - logp_likelihood[0]

    def _predict_output_cell(self, rowid, cout, label):
        output_idx = self.outputs.index(cout)
        sp_rowid = '(atom %d)' % (rowid,)
        return self.ripl.predict('((lookup outputs %i) %s)'
            % (output_idx, sp_rowid), label=label)

    def _observe_output_cell(self, rowid, cout, value):
        if rowid not in self.labels:
            self.labels[rowid] = dict()
        output_idx = self.outputs.index(cout)
        label = self._gen_label()
        sp_rowid = '(atom %d)' % (rowid,)
        if not self.observe_custom:
            self.ripl.observe('((lookup outputs %i) %s)'
                % (output_idx, sp_rowid), value, label=label)
        else:
            obs_args = '%s %s (quote %s)' % (sp_rowid, value, label)
            self.ripl.evaluate('((lookup observers %i) %s)'
                % (output_idx, obs_args))
        self.labels[rowid][cout] = label

    def _observe_input_cell(self, rowid, cin, value):
        input_name = self.input_mapping[cin]
        sp_rowid = '(atom %d)' % (rowid,)
        input_cell_name = self._get_input_cell_name(rowid, cin)
        self.ripl.execute_program(
            '(assume %s (dict_set (lookup inputs "%s") %s %s))'
            % (input_cell_name, input_name, sp_rowid, value))

    def _forget_output_cell(self, rowid, cout):
        label = self.labels[rowid][cout]
        self.ripl.forget(label)
        del self.labels[rowid][cout]
        if len(self.labels[rowid]) == 0:
            del self.labels[rowid]

    def _forget_input_cell(self, rowid, cin):
        input_name = self.input_mapping[cin]
        sp_rowid = '(atom %d)' % (rowid,)
        self.ripl.sample('(dict_pop (lookup inputs "%s") %s)'
            % (input_name, sp_rowid))
        # Forget the assume directive.
        input_cell_name = self._get_input_cell_name(rowid, cin)
        self.ripl.forget(input_cell_name)

    def _is_observed_output_cell(self, rowid, cout):
        return rowid in self.labels and cout in self.labels[rowid]

    def _is_observed_input_cell(self, rowid, cin):
        input_name = self.input_mapping[cin]
        sp_rowid = '(atom %d)' % (rowid,)
        return self.ripl.sample('(contains (lookup inputs "%s") %s)'
            % (input_name, sp_rowid))

    def _get_input_cell_value(self, rowid, cin):
        input_name = self.input_mapping[cin]
        sp_rowid = '(atom %d)' % (rowid,)
        return self.ripl.sample('(lookup (lookup inputs "%s") %s)'
            % (input_name, sp_rowid))

    def _get_input_cell_name(self, rowid, cin):
        str_rowid = '%s%s' % ('' if 0 <= rowid else 'm', abs(rowid))
        return '%s_%s' % (self.input_mapping[cin], str_rowid)

    def _gen_label(self):
        return 't%s%s' % (
            self.rng.randint(1,100),
            datetime.now().strftime('%Y%m%d%H%M%S%f'))

    # Validating observations, targets, and constraints.

    def _cleanse_constraints(self, rowid, targets, constraints):
        constraints = constraints or {}
        if any(math.isnan(value) for value in constraints.itervalues()):
            raise ValueError('Nan constraints: %s' % (constraints,))
        if not all(cout in self.outputs for cout in constraints):
            raise ValueError('Unknown constraints: %s' % (constraints,))
        if set.intersection(set(targets), set(constraints)):
            raise ValueError('Overlapping targets and constraints: %s, %s'
                % (targets, constraints,))
        constraints_obs = [cout for cout in constraints
            if self._is_observed_output_cell(rowid, cout)]
        if constraints_obs:
            raise ValueError('Constrained output already observed: %d, %s, %s'
                % (rowid, constraints, constraints_obs))
        return constraints

    def _cleanse_inputs(self, rowid, inputs):
        inputs = inputs or {}
        if any(math.isnan(value) for value in inputs.itervalues()):
            raise ValueError('Nan inputs: %s' % inputs)
        if not all(cin in self.inputs for cin in inputs):
            raise ValueError('Unknown inputs: %s' % (inputs,))
        inputs_obs = set(cin for cin in inputs
            if self._is_observed_input_cell(rowid, cin))
        inputs_vals = [(self._get_input_cell_value(rowid, cin), inputs[cin])
            for cin in inputs_obs]
        if any(gu.abserr(v1, v2) > 1e-6 for (v1, v2) in inputs_vals):
            raise ValueError('Given inputs contradict dataset: %d, %s, %s, %s'
                % (rowid, inputs, inputs_obs, inputs_vals))
        return {cin : inputs[cin] for cin in inputs if cin not in inputs_obs}

    def _cleanse_observation(self, rowid, observation):
        if not observation:
            raise ValueError('No observation: %s.' % observation)
        if not set.issubset(set(observation), set(self.outputs)):
            raise ValueError('Unknown observation: %s,%s'
                % (observation, self.outputs))
        if any(math.isnan(value) for value in observation.itervalues()):
            raise ValueError('Nan observation: %s' % (observation,))
        if rowid in self.labels \
                and any(cout in self.labels[rowid] for cout in observation):
            raise ValueError('Observation exists: %d %s' % (rowid, observation))
        return observation

    def _cleanse_targets(self, rowid, targets):
        if not targets:
            raise ValueError('No targets: %s' % (targets,))
        if not all(cout in self.outputs for cout in targets):
            raise ValueError('Unknown targets: %s' % (targets,))
        if isinstance(targets, dict):
            if any(math.isnan(value) for value in targets.itervalues()):
                raise ValueError('Nan targets: %s' % (targets,))
            targets_obs = [cout for cout in targets
                if self._is_observed_output_cell(rowid, cout)]
            if targets_obs:
                raise ValueError('Constrained output already observed: '
                    '%d, %s, %s' % (rowid, targets, targets_obs))
        return targets

    def _get_num_observers(self):
        # Return the length of the "observers" list defined by the client, or
        # None if the client did not override the observers.
        try:
            return self.ripl.evaluate('(size observers)')
        except VentureException:
            return None

    def _get_input_mapping(self, inputs):
        # Return mapping from input integer index to string name.
        input_dict = self.ripl.sample('inputs')
        assert len(inputs) == len(input_dict)
        return {cin:e[0] for e, cin in zip(input_dict, inputs)}

    def _check_input_args(self, rowid, inputs):
        inputs_obs = set(cin for cin in inputs
            if self._is_observed_input_cell(rowid, cin))
        inputs_vals = [(self._get_input_cell_value(rowid, cin), inputs[cin])
            for cin in inputs_obs]
        if any(gu.abserr(v1, v2) > 1e-6 for (v1, v2) in inputs_vals):
            raise ValueError('Given inputs contradict dataset: %d, %s, %s, %s'
                % (rowid, inputs, inputs_obs, inputs_vals))
        return {cin : inputs[cin] for cin in inputs if cin not in inputs_obs}

    @staticmethod
    def convert_key_int_to_str(d):
        assert all(isinstance(c, int) for c in d)
        return {str(c): v for c, v in d.iteritems()}

    @staticmethod
    def convert_key_str_to_int(d):
        assert all(isinstance(c, (str, unicode)) for c in d)
        return {int(c): v for c, v in d.iteritems()}

    @staticmethod
    def _load_helpers(ripl):
        ripl._compute_search_paths(
                [os.path.abspath(os.path.dirname(__file__))])
        ripl.load_plugin('helpers.py')
        return ripl
