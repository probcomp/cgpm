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
from scipy.sparse.csgraph import connected_components


def validate_cgpms(cgpms):
    ot = [set(c.outputs) for c in cgpms]
    if not all(s for s in ot):
        raise ValueError('No output for a cgpm: %s' % ot)
    if any(set.intersection(a,b) for a,b in it.combinations(ot, 2)):
        raise ValueError('Duplicate outputs for cgpms: %s' % ot)
    return cgpms


def retrieve_variable_to_cgpm(cgpms):
    """Return map of variable v to its index i in the list of cgpms."""
    return {v:i for i, c in enumerate(cgpms) for v in c.outputs}


def retrieve_adjacency_list(cgpms, v_to_c):
    """Return map of cgpm index to list of indexes of its parent cgpms."""
    return {
        i: list(set(v_to_c[p] for p in c.inputs if p in v_to_c))
        for i, c in enumerate(cgpms)
    }

def retrieve_adjacency_matrix(cgpms, v_to_c):
    """Return a directed adjacency matrix of cgpms."""
    adjacency_list = retrieve_adjacency_list(cgpms, v_to_c)
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)))
    for i in adjacency_list:
        adjacency_matrix[i, adjacency_list[i]] = 1
    return adjacency_matrix.T

def retrieve_extraneous_inputs(cgpms, v_to_c):
    """Return list of inputs that are not the output of any cgpm."""
    extraneous = [[i for i in c.inputs if i not in v_to_c] for c in cgpms]
    return list(it.chain.from_iterable(extraneous))


def retrieve_ancestors(cgpms, q):
    """Return list of all variables that are ancestors of q (duplicates)."""
    v_to_c = retrieve_variable_to_cgpm(cgpms)
    if q not in v_to_c:
        raise ValueError('Invalid node: %s, %s' % (q, v_to_c))
    def ancestors(v):
        parents = cgpms[v_to_c[v]].inputs if v in v_to_c else []
        parent_ancestors = [ancestors(v) for v in parents]
        return list(it.chain.from_iterable(parent_ancestors)) + parents
    return ancestors(q)

def retrieve_descendents(cgpms, q):
    """Return list of all variables that are descends of q (duplicates)."""
    v_to_c = retrieve_variable_to_cgpm(cgpms)
    if q not in v_to_c:
        raise ValueError('Invalid node: %s, %s' % (q, v_to_c))
    def descendents(v):
        children = list(it.chain.from_iterable(
            [c.outputs for c in cgpms if v in c.inputs]))
        children_descendents = [descendents(c) for c in children]
        return list(it.chain.from_iterable(children_descendents)) + children
    return descendents(q)


def retrieve_weakly_connected_components(cgpms):
    v_to_c = retrieve_variable_to_cgpm(cgpms)
    adjacency = retrieve_adjacency_matrix(cgpms, v_to_c)
    n_components, labels = connected_components(
        adjacency, directed=True, connection='weak', return_labels=True)
    return labels


def topological_sort(graph):
    """Topologically sort a directed graph represented as an adjacency list.

    Assumes that edges are incoming, ie (10: [8,7]) means 8->10 and 7->10.

    Parameters
    ----------
    graph : list or dict
        Adjacency list or dict representing the graph, for example:
            graph_l = [(10, [8, 7]), (5, [8, 7, 9, 10, 11, 13, 15])]
            graph_d = {10: [8, 7], 5: [8, 7, 9, 10, 11, 13, 15]}

    Returns
    -------
    graph_sorted : list
        An adjacency list, where order of nodes is in topological order.
    """
    graph_sorted = []
    graph = dict(graph)
    while graph:
        cyclic = True
        for node, edges in list(graph.items()):
            if all(e not in graph for e in edges):
                cyclic = False
                del graph[node]
                graph_sorted.append(node)
        if cyclic:
            raise ValueError('Cyclic dependency occurred in topological_sort.')
    return graph_sorted


def retrieve_required_inputs(cgpms, topo, targets, constraints, extraneous):
    """Return list of input addresses required to answer query."""
    required = set(targets)
    for l in reversed(topo):
        outputs_l = cgpms[l].outputs
        inputs_l = cgpms[l].inputs
        if any(i in required or i in constraints for i in outputs_l):
            required.update(inputs_l)
    return [
        target for target in required if
        all(target not in x for x in [targets, constraints, extraneous])
    ]
