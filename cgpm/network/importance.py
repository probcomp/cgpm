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

from builtins import zip
from builtins import range
from builtins import object
import itertools

from math import isinf

from cgpm.network import helpers as hu
from cgpm.utils import general as gu


class ImportanceNetwork(object):
    """Querier for a Composite CGpm."""

    def __init__(self, cgpms, accuracy=None, rng=None):
        if accuracy is None:
            accuracy = 1
        self.rng = rng if rng else gu.gen_rng(1)
        self.cgpms = hu.validate_cgpms(cgpms)
        self.accuracy = accuracy
        self.v_to_c = hu.retrieve_variable_to_cgpm(self.cgpms)
        self.adjacency = hu.retrieve_adjacency_list(self.cgpms, self.v_to_c)
        self.extraneous = hu.retrieve_extraneous_inputs(self.cgpms, self.v_to_c)
        self.topo = hu.topological_sort(self.adjacency)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        if constraints is None:
            constraints = {}
        if inputs is None:
            inputs = {}
        samples, weights = list(zip(*[
            self.weighted_sample(rowid, targets, constraints, inputs)
            for _i in range(self.accuracy)
        ]))
        if all(isinf(l) for l in weights):
            raise ValueError('Zero density constraints: %s' % (constraints,))
        # Skip an expensive random choice if there is only one option.
        index = 0 if self.accuracy == 1 else \
            gu.log_pflip(weights, rng=self.rng)
        return {q: samples[index][q] for q in targets}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        if constraints is None:
            constraints = {}
        if inputs is None:
            inputs = {}
        # Compute joint probability.
        samples_joint, weights_joint = list(zip(*[
            self.weighted_sample(
                rowid, [], gu.merged(targets, constraints), inputs)
            for _i in range(self.accuracy)
        ]))
        logp_joint = gu.logmeanexp(weights_joint)
        # Compute marginal probability.
        samples_marginal, weights_marginal = list(zip(*[
            self.weighted_sample(rowid, [], constraints, inputs)
            for _i in range(self.accuracy)
        ])) if constraints else ({}, [0.])
        if all(isinf(l) for l in weights_marginal):
            raise ValueError('Zero density constraints: %s' % (constraints,))
        logp_constraints = gu.logmeanexp(weights_marginal)
        # Return log ratio.
        return logp_joint - logp_constraints

    def weighted_sample(self, rowid, targets, constraints, inputs):
        targets_required = self.retrieve_required_inputs(targets, constraints)
        targets_all = targets + targets_required
        sample = dict(constraints)
        weight = 0
        for l in self.topo:
            sl, wl = self.invoke_cgpm(
                rowid, self.cgpms[l], targets_all, sample, inputs)
            sample.update(sl)
            weight += wl
        assert set(sample) == set.union(set(constraints), set(targets_all))
        return sample, weight

    def invoke_cgpm(self, rowid, cgpm, targets, constraints, inputs):
        cgpm_inputs = {
            e : x for e, x in
                itertools.chain(iter(inputs.items()), iter(constraints.items()))
            if e in cgpm.inputs
        }
        cgpm_constraints = {
            e:x for e, x in constraints.items()
            if e in cgpm.outputs
        }
        # ev_all = gu.merged(ev_in, ev_out)
        cgpm_targets = [q for q in targets if q in cgpm.outputs]
        if cgpm_constraints or cgpm_targets:
            assert all(i in cgpm_inputs for i in cgpm.inputs)
        weight = cgpm.logpdf(
            rowid,
            targets=cgpm_constraints,
            constraints=None,
            inputs=cgpm_inputs) if cgpm_constraints else 0
        sample = cgpm.simulate(
            rowid,
            targets=cgpm_targets,
            constraints=cgpm_constraints,
            inputs=cgpm_inputs
            ) if cgpm_targets else {}
        return sample, weight

    def retrieve_required_inputs(self, targets, constraints):
        """Return list of inputs required to answer query."""
        def retrieve_required_inputs(cgpm, targets):
            active = any(i in targets or i in constraints for i in cgpm.outputs)
            return cgpm.inputs if active else []
        required_all = set(targets)
        for l in reversed(self.topo):
            required_l = retrieve_required_inputs(self.cgpms[l], required_all)
            required_all.update(required_l)
        return [c for c in required_all if
            all(c not in x for x in [targets, constraints, self.extraneous])]
