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

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        if evidence is None: evidence = {}
        samples, weights = zip(*[
            self.weighted_sample(rowid, query, evidence)
            for i in xrange(self.accuracy)
        ])
        if all(isinf(l) for l in weights):
            raise ValueError('Zero density evidence: %s' % (evidence))
        # Skip an expensive random choice if there is only one option.
        index = 0 if self.accuracy == 1 else \
            gu.log_pflip(weights, rng=self.rng)
        return {q: samples[index][q] for q in query}

    def logpdf(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        # Compute joint probability.
        samples_joint, weights_joint = zip(*[
            self.weighted_sample(rowid, [], gu.merged(evidence, query))
            for i in xrange(self.accuracy)
        ])
        logp_joint = gu.logmeanexp(weights_joint)
        # Compute marginal probability.
        samples_marginal, weights_marginal = zip(*[
            self.weighted_sample(rowid, [], evidence)
            for i in xrange(self.accuracy)
        ]) if evidence else ({}, [0.])
        if all(isinf(l) for l in weights_marginal):
            raise ValueError('Zero density evidence: %s' % (evidence))
        logp_evidence = gu.logmeanexp(weights_marginal)
        # Return log ratio.
        return logp_joint - logp_evidence

    def weighted_sample(self, rowid, query, evidence):
        query_all = query + self.retrieve_missing_inputs(query, evidence)
        sample = dict(evidence)
        weight = 0
        for l in self.topo:
            sl, wl = self.invoke_cgpm(rowid, self.cgpms[l], query_all, sample)
            sample.update(sl)
            weight += wl
        assert set(sample.keys()) == set.union(set(evidence), set(query_all))
        return sample, weight

    def invoke_cgpm(self, rowid, cgpm, query, evidence):
        ev_in = {e:x for e,x in evidence.iteritems() if e in cgpm.inputs}
        ev_out = {e:x for e,x in evidence.iteritems() if e in cgpm.outputs}
        ev_all = gu.merged(ev_in, ev_out)
        qry_out = [q for q in query if q in cgpm.outputs]
        if ev_out or qry_out: assert all(i in evidence for i in cgpm.inputs)
        weight = cgpm.logpdf(rowid, ev_out, ev_in) if ev_out else 0
        sample = cgpm.simulate(rowid, qry_out, ev_all) if qry_out else {}
        return sample, weight

    def retrieve_missing_inputs(self, query, evidence):
        """Return list of inputs (not in evidence) required to answer query."""
        def retrieve_missing_input(cgpm, query):
            active = any(i in query or i in evidence for i in cgpm.outputs)
            return cgpm.inputs if active else []
        missing = set(query)
        for l in reversed(self.topo):
            missing_l = retrieve_missing_input(self.cgpms[l], missing)
            missing.update(missing_l)
        return [c for c in missing if
            all(c not in x for x in [query, evidence, self.extraneous])]
