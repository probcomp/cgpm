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

import itertools as it

import numpy as np


def validate_crp_constrained_partition(Zv, Cd, Ci, Rd, Ri):
    """Only tests the outer CRP partition Zv."""
    valid = True
    N = len(Zv)
    for block in Cd:
        valid = valid and all(Zv[block[0]] == Zv[b] for b in block)
        for a, b in it.combinations(block, 2):
            valid = valid and check_compatible_customers(Cd, Ci, Ri, Rd, a, b)
    for a, b in Ci:
        valid = valid and not Zv[a] == Zv[b]
    return valid

def validate_dependency_constraints(N, Cd, Ci):
    """Validates Cd and Ci constraints on N columns."""
    # Allow unknown number of customers.
    if N is None:
        N = 1e10
    counts = {}
    for block in Cd:
        # Every constraint must be more than one customer.
        if len(block) == 1:
            raise ValueError('Single customer in dependency constraint.')
        for col in block:
            # Every column must have correct index.
            if N <= col:
                raise ValueError('Dependence customer out of range.')
            # Every column must appear once only.
            if col not in counts:
                counts[col] = 0
            counts[col] += 1
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

def check_compatible_constraints(Cd1, Ci1, Cd2, Ci2):
    """Returns True if (Cd1, Ci1) is compatible with (Cd2, Ci2)."""
    try:
        validate_dependency_constraints(None, Cd1, Ci1)
        validate_dependency_constraints(None, Cd1, Ci2)
        validate_dependency_constraints(None, Cd2, Ci1)
        validate_dependency_constraints(None, Cd2, Ci2)
        return True
    except ValueError:
        return False

def check_compatible_customers(Cd, Ci, Ri, Rd, a, b):
    """Checks if customers a,b are compatible."""
    # Explicitly independent.
    if (a,b) in Ci or (b,a) in Ci:
        return False
    # Incompatible Rd/Ri constraints.
    if (a in Rd or a in Ri) and (b in Rd or b in Ri):
        return check_compatible_constraints(
            Rd.get(a,[]), Ri.get(a,[]), Rd.get(b,[]), Ri.get(b,[]))
    return True

def validate_crp_constrained_input(N, Cd, Ci, Rd, Ri):
    # First validate outer Cd, Ci.
    validate_dependency_constraints(N, Cd, Ci)
    # Validate all inner Rd, Ri.
    for c in Rd:
        col_dep = Rd[c]
        row_dep = Ri.get(c,{})
        validate_dependency_constraints(None, col_dep, row_dep)
    # For each block in Cd, validate their Rd, Ri are compatible.
    for block in Cd:
        for a,b in it.combinations(block, 2):
            if not check_compatible_customers(Cd, Ci, Ri, Rd, a, b):
                raise ValueError('Incompatible row constraints for dep cols.')
    return True

def validate_query_evidence(X, rowid, hypothetical, query, evidence=None):
    if evidence is None: evidence = {}
    # Simulate or logpdf query?
    simulate = isinstance(query, list)
    # Disallow duplicated query cols.
    if simulate and len(set(query)) != len(query):
        raise ValueError('Query columns must be unique.')
    # Disallow overlap between query and evidence.
    if len(set.intersection(set(query), set(evidence))) > 0:
        raise ValueError('Query and evidence columns must be disjoint.')
    # Skip rest.
    # Disallow evidence overriding non-nan cells.
    if not hypothetical and any(not np.isnan(X[e][rowid]) for e in evidence):
        raise ValueError('Cannot evidence a non-nan observed cell.')
    # XXX DETERMINE ME!
    # Disallow query of observed cell. It is already observed so Dirac.
    # if (not hypothetical
    #         and not simulate and any(not np.isnan(X[q][rowid]) for q in query)):
    #     raise ValueError('Cannot query a non-nan observed cell.')

def partition_query_evidence(Z, query, evidence):
    """Returns queries[k], evidences[k] are queries, evidences for cluster k."""
    evidences = partition_dict(Z, evidence)
    if isinstance(query, list):
        queries = partition_list(Z, query)
    else:
        queries = partition_dict(Z, query)
    return queries, evidences

def partition_list(Z, L):
    result = {}
    for l in L:
        k = Z[l]
        if k in result:
            result[k].append(l)
        else:
            result[k] = [l]
    return result

def partition_dict(Z, L):
    result = {}
    for l in L:
        k, val = Z[l], (l, L[l])
        if k in result:
            result[k].append(val)
        else:
            result[k] = [val]
    return {k: dict(v) for k,v in result.iteritems()}
