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

import numpy as np

def validate_crp_constrained_partition(Zv, Cd, Ci):
    valid = True
    for block in Cd:
        valid = valid and all(Zv[block[0]] == Zv[b] for b in block)
    for a, b in Ci:
        valid = valid and not Zv[a] == Zv[b]
    return valid

def validate_crp_constrained_input(N, Cd, Ci):
    """Validates Cd and Ci constraints on N columns."""
    counts = [0]*N
    for block in Cd:
        # Every constraint must be more than column.
        if len(block) == 1:
            raise ValueError('Single customer in dependency constraint.')
        for col in block:
            # Every column must have correct index.
            if N <= col:
                raise ValueError('Dependence customer out of range.')
            counts[col] += 1
            # Every column must appear once only.
            if counts[col] > 1:
                raise ValueError('Multiple customer dependencies.')
        for pair in Ci:
            # Ci cannot include columns in same Cd block.
            if pair[0] in block and pair[1] in block:
                raise ValueError('Contradictory customer independence.')
    for pair in Ci:
        # Ci entries are tuples only.
        if len(pair) != 2:
            raise ValueError('Independencies require two customers.')
        if N <= pair[0] or N <= pair[1]:
            raise ValueError('Independence customer of out range.')
        # Dummy case.
        if pair[0] == pair[1]:
            raise ValueError('Independency specified for same customer.')
    return True

def validate_query_evidence(X, rowid, hypothetical, query, evidence=None):
    if evidence is None:
        evidence = []
    qcols = [q[0] for q in query] if isinstance(query[0], list) else query
    ecols = [e[0] for e in evidence]
    # Disallow duplicated query cols.
    if len(set(qcols)) != len(qcols):
        raise ValueError('Query columns must be unique.')
    # Disallow overlap between query and evidence.
    if len(set.intersection(set(qcols), set(ecols))) > 0:
        raise ValueError('Query and evidence columns must be disjoint.')
    # Skip rest.
    if hypothetical:
        return
    # Disallow evidence overriding non-nan cells.
    if any(not np.isnan(X[rowid,ec]) for ec in ecols):
        raise ValueError('Cannot evidence a non-nan observed cell.')
    # XXX DISABLED
    # Disallow query of observed cell. It is already observed so Dirac.
    # if any(not np.isnan(X[rowid,ec]) for ec in ecols):
    #     raise ValueError('Cannot query a non-nan observed cell.')
