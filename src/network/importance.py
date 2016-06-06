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

    def weighted_sample(self, r, evidence):
        pass

    def simulate_layer(self, r, l, s):
        # Layer is self.cgpms[l], and s is the aggregated sample.
        cgpm = self.cgpms[l]
        evidence = {e:x for e,x in s if e in cgpm.inputs}
        query = {q:x for e,x in s if q in cgpm.outputs}
        logp = cgpm.logpdf(r, query, evidence) if query else 0
        evidence = gu.merge_dicts(query, evidence)
