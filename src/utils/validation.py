# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

def validate_dependency_constraints(Zv, Cd=None, Ci=None):
    if Cd is None:
        Cd = []
    if Ci is None:
        Ci = []
    valid = True
    for block in Cd:
        valid = valid and all(Zv[block[0]] == Zv[b] for b in block)
    for a, b in Ci:
        valid = valid and all(not Zv[a] == Zv[b])
    return valid

def validate_dependency_input(N, Cd=None, Ci=None):
    """Validates Cd and Ci constraints on N columns."""
    if Ci is None:
        Ci = []
    if Cd is None:
        Cd = []
    counts = [0]*N
    for block in Cd:
        # Every constraint must be more than column.
        if len(block) == 1:
            raise ValueError('Single column in dependency constraint.')
        for col in block:
            # Every column must have correct index.
            if N <= col:
                raise ValueError('Dependence column out of range.')
            counts[col] += 1
            # Every column must appear once only.
            if counts[col] > 1:
                raise ValueError('Multiple column dependencies.')
        for pair in Ci:
            # Ci cannot include columns in same Cd block.
            if pair[0] in block and pair[1] in block:
                raise ValueError('Contradictory column independence.')
    for pair in Ci:
        # Ci entries are tuples only.
        if len(pair) != 2:
            raise ValueError('Independencies require two columns.')
        if N <= pair[0] or N <= pair[1]:
            raise ValueError('Independence column of out range.')
        # Dummy case.
        if pair[0] == pair[1]:
            raise ValueError('Independency specified for same column.')

def validate_query_evidence(X, rowid, hypothetical, query, evidence=None):
    if evidence is None:
        evidence = []
    qcols = [q[0] for q in query] if isinstance(query[0], list) else query
    ecols = [e[0] for e in evidence]
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
