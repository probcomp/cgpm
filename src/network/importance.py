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

import gpmcc.network.helpers as hu
import gpmcc.utils.general as gu


class ImportanceNetwork(object):
    """Querier for a Composite CGpm."""

    def __init__(self, cgpms, accuracy=1, rng=None):
        self.rng = rng if rng else gu.gen_rng(1)
        self.cgpms = hu.validate_cgpms(cgpms)
        self.accuracy = accuracy
        self.v_to_c = hu.retrieve_variable_to_cgpm(self.cgpms)
        self.adjacency = hu.retrieve_adjacency(self.cgpms, self.v_to_c)
        self.extraneous = hu.retrieve_extraneous_inputs(self.cgpms, self.v_to_c)
        self.topo = hu.topological_sort(self.adjacency)

    def simulate(self, rowid, query, evidence=None, N=1):
        if evidence is None: evidence = {}
        samples, weights = zip(*
            [self.weighted_sample(rowid, query, evidence)
            for i in xrange(self.accuracy*N)])
        indices = gu.log_pflip(weights, size=N, rng=self.rng)
        return [{q: samples[i][q] for q in query} for i in indices]

    def logpdf(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        # Compute joint probability.
        samples_joint, weights_joint = zip(*
            [self.weighted_sample(rowid, [], gu.merged(evidence, query))
            for i in xrange(self.accuracy)])
        logp_joint = gu.logmeanexp(weights_joint)
        # Compute marginal probability.
        samples_marginal, weights_marginal = zip(*
            [self.weighted_sample(rowid, [], evidence)
            for i in xrange(self.accuracy)]) if evidence else ({}, [0.])
        logp_evidence = gu.logmeanexp(weights_marginal)
        # Take log ratio.
        return logp_joint - logp_evidence

    def weighted_sample(self, rowid, query, evidence):
        query_all = query + self.retrieve_missing_inputs(query, evidence)
        sample = dict(evidence)
        weight = 0
        for l in self.topo:
            sl, wl = self.invoke_cgpm(rowid, self.cgpms[l], query_all, sample)
            sample.update(sl)
            weight += wl
        assert all(q in sample for q in sample)
        return sample, weight

    def invoke_cgpm(self, rowid, cgpm, query, evidence):
        assert all(i in evidence for i in cgpm.inputs)
        ev_in = {e:x for e,x in evidence.items() if e in cgpm.inputs}
        ev_out = {e:x for e,x in evidence.items() if e in cgpm.outputs}
        ev_all = gu.merged(ev_in, ev_out)
        qry_out = [q for q in query if q in cgpm.outputs]
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
