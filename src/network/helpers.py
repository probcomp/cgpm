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


def retrieve_adjacency(cgpms, v_to_c):
    """Return map of cgpm index to list of indexes of its parent cgpms."""
    return {i: list(set(v_to_c[p] for p in c.inputs if p in v_to_c))
        for i, c in enumerate(cgpms)}


def retrieve_extraneous_inputs(cgpms, v_to_c):
    """Return list of inputs that are not the output of any cgpm."""
    extraneous = [[i for i in c.inputs if i not in v_to_c] for c in cgpms]
    return list(it.chain.from_iterable(extraneous))


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
        for node, edges in graph.items():
            if all(e not in graph for e in edges):
                cyclic = False
                del graph[node]
                graph_sorted.append(node)
        if cyclic:
            raise ValueError('Cyclic dependency occurred in topological_sort.')
    return graph_sorted
