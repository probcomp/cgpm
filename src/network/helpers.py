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

import itertools


def validate_cgpms(cgpms):
    ot = [set(c.outputs) for c in cgpms]
    if not all(s for s in ot):
        raise ValueError('Not output for a cgpm: %s' % ot)
    if any(set.intersection(a,b) for a,b in itertools.combinations(ot, 2)):
        raise ValueError('Duplicate outputs for cgpms: %s' % ot)
    return cgpms


def retrieve_variable_to_cgpm(cgpms):
    # Returns map of variable to its index in the cgpms list.
    return {v:i for i, c in enumerate(cgpms) for v in c.outputs}


def retrieve_adjacency(cgpms, v_to_c):
    # v_to_c is the output of retrieve_variable_to_cgpm
    return {i: list(set(v_to_c[p] for p in c.inputs if p in v_to_c))
        for i,c in enumerate(cgpms)}


def retrieve_extranous_inputs(cgpms, v_to_c):
    # v_to_c is the output of retrieve_variable_to_cgpm
    missing = [[i for i in c.inputs if i not in v_to_c] for c in cgpms]
    return [item for m in missing for item in m]


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
        acyclic = False
        for node, edges in graph.items():
            for edge in edges:
                if edge in graph:
                    break
            else:
                acyclic = True
                del graph[node]
                graph_sorted.append((node, edges))
        if not acyclic:
            raise ValueError('A cyclic dependency occurred in topological_sort')
    return graph_sorted
