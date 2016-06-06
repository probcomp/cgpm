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
    """The outer most GPM in gpmcc."""

    def __init__(self, X, cgpms, rng=None):
        self.rng = rng if rng else gu.gen_rng(1)
        self.X = X
        self.cgpms = hu.validate_cgpms(cgpms)
        self.v_to_c = hu.retrieve_variable_to_cgpm(self.cgpms)
        self.adjacency = hu.retrieve_adjacency(self.cgpms, self.v_to_c)
        self.extraneous = hu.retrieve_extraneous_inputs(self.cgpms, self.v_to_c)
        self.topo = hu.topological_sort(self.adjacency)

    def simulate(self, r, query, evidence):
        pass

    def logpdf(self, r, query, evidence):
        pass

    def weighted_sample(self, r, query, evidence):
        pass

    def invoke_cgpm(self, r, index, query, evidence):
        cgpm = self.cgpms[index]
        assert isinstance(query, list)
        assert all(i in evidence for i in cgpm.inputs)
        assert isinstance(evidence, dict)
        assert all(q in cgpm.outputs for q in query)
        assert all(e in cgpm.inputs or e in cgpm.outputs for e in evidence)
        ev_in = {e:x for e,x in evidence.items() if e in cgpm.inputs}
        ev_out = {e:x for e,x in evidence.items() if e in cgpm.outputs}
        weight = cgpm.logpdf(r, ev_out, ev_in) if ev_out else 0
        sample = cgpm.simulate(r, query, evidence)
        return sample, weight


    def retrieve_missing_inputs(self, query, evidence):
        """Return list of inputs (not in evidence) required to answer query."""

        def retrieve_missing_input(cgpm, query):
            active = any(i in query or i in evidence for i in cgpm.outputs)
            return cgpm.inputs if active else []

        missing = set(query)
        for l in reversed(self.topo):
            missing_l = retrieve_missing_input(self.cgpms[l[0]], missing)
            missing.update(missing_l)

        return [c for c in missing if
            all(c not in x for x in [query, evidence, self.extraneous])]
