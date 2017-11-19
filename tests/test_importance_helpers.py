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

from collections import namedtuple

import numpy as np
import pytest

from cgpm.network import helpers
from cgpm.network.importance import ImportanceNetwork


CGpm = namedtuple('CGpm', ['outputs', 'inputs'])


def build_cgpm_no_connection():
    return [
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[1], inputs=[]),
        CGpm(outputs=[8], inputs=[]),
    ]


def build_cgpms_v_structure():
    return [
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[1], inputs=[]),
        CGpm(outputs=[8], inputs=[2, 1]),
    ]


def build_cgpms_markov_chain():
    return [
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[1], inputs=[8]),
        CGpm(outputs=[8], inputs=[2, 5]),
    ]


def build_cgpms_complex():
    return [
        CGpm(outputs=[2, 14], inputs=[4, 5, -8]),
        CGpm(outputs=[3, 15], inputs=[4, -9]),
        CGpm(outputs=[5], inputs=[0, -10, -11]),
        CGpm(outputs=[4, 16], inputs=[5, -12]),
    ]


def build_cgpms_fork():
    return [
        CGpm(outputs=[0], inputs=[1,2]),
        CGpm(outputs=[1], inputs=[3]),
        CGpm(outputs=[2], inputs=[4]),
    ]


def build_cgpms_four_forests():
    return [
        # First component
        CGpm(outputs=[2, 14], inputs=[4, 5, -8]),
        CGpm(outputs=[3, 15], inputs=[4, -9]),
        CGpm(outputs=[5], inputs=[0, -10, -11]),
        CGpm(outputs=[4, 16], inputs=[5, -12]),
        # Second component.
        CGpm(outputs=[1000], inputs=[1001,1002]),
        CGpm(outputs=[1001], inputs=[1003]),
        CGpm(outputs=[1002], inputs=[1004]),
        # Third component.
        CGpm(outputs=[2002], inputs=[]),
        CGpm(outputs=[2001], inputs=[]),
        CGpm(outputs=[2008], inputs=[2002, 2001]),
        # Fourth component.
        CGpm(outputs=[30002], inputs=[]),
        CGpm(outputs=[30001], inputs=[30008]),
        CGpm(outputs=[30008], inputs=[30002, 30005]),
    ]


def test_retrieve_variable_to_cgpm():
    cgpms = [
        CGpm(outputs=[0, 1, 5], inputs=[2]),
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[-1], inputs=[7]),
    ]
    for order in itertools.permutations(cgpms):
        variable_to_cgpm = helpers.retrieve_variable_to_cgpm(order)
        for v, c in variable_to_cgpm.iteritems():
            assert v in order[c].outputs

def test_retrieve_adjacency_list():
    # No connections.
    cgpms = build_cgpm_no_connection()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_list(cgpms, vtc)
    assert {0: [], 1:[], 2:[]} == adj

    # V structure.
    cgpms = build_cgpms_v_structure()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_list(cgpms, vtc)
    assert {0: [], 1:[], 2:[0, 1]} == adj

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_list(cgpms, vtc)
    assert {0: [], 1:[2], 2:[0]} == adj

    # Complex.
    cgpms = build_cgpms_complex()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_list(cgpms, vtc)
    assert {0: [2,3], 1:[3], 2:[], 3:[2]} == adj

def test_retrieve_adjacency_matrix():
    # No connections.
    cgpms = build_cgpm_no_connection()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_matrix(cgpms, vtc)
    assert np.allclose(adj, np.asarray([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]))

    # V structure.
    cgpms = build_cgpms_v_structure()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_matrix(cgpms, vtc)
    assert np.allclose(adj, np.asarray([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]))

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_matrix(cgpms, vtc)
    assert np.allclose(adj, np.asarray([
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
    ]))

    # Complex.
    cgpms = build_cgpms_complex()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_matrix(cgpms, vtc)
    assert np.allclose(adj, np.asarray([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
    ]))

    # Fork.
    cgpms = build_cgpms_fork()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency_matrix(cgpms, vtc)
    assert np.allclose(adj, np.asarray([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]))


def test_weakly_connected_components():
    # No connections.
    cgpms = build_cgpm_no_connection()
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 1, 2]
    )

    # V structure.
    cgpms = build_cgpms_v_structure()
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0]
    )

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0]
    )

    # Complex.
    cgpms = build_cgpms_complex()
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0, 0]
    )

    # Fork.
    cgpms = build_cgpms_fork()
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0]
    )

    # Four forests.
    cgpms = build_cgpms_four_forests()
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    )

    # Three forests, merging the first and second.
    cgpms = build_cgpms_four_forests()
    cgpms[4] = CGpm(outputs=[1000], inputs=[1001,1002,14])
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    )

    # Two forests, merging the first and fourth, and second and third.
    cgpms = build_cgpms_four_forests()
    cgpms[4] = CGpm(outputs=[1000], inputs=[1001,1002,2008])
    cgpms[0] = CGpm(outputs=[2, 14], inputs=[4, 5, -8, 30008])
    assert np.allclose(
        helpers.retrieve_weakly_connected_components(cgpms),
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    )


def test_retrieve_extraneous_inputs():
    # No connections.
    cgpms = build_cgpm_no_connection()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    ext = helpers.retrieve_extraneous_inputs(cgpms, vtc)
    assert [] == ext

    # V structure.
    cgpms = build_cgpms_v_structure()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    ext = helpers.retrieve_extraneous_inputs(cgpms, vtc)
    assert [] == ext

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    ext = helpers.retrieve_extraneous_inputs(cgpms, vtc)
    assert [5] == ext

    # Complex.
    cgpms = build_cgpms_complex()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    ext = helpers.retrieve_extraneous_inputs(cgpms, vtc)
    assert set([0, -8, -9, -10, -11, -12]) == set(ext)


def test_retrieve_required_inputs():
    # No connections.
    network = ImportanceNetwork(build_cgpm_no_connection())
    missing = network.retrieve_required_inputs([2], {8: None})
    assert [] == missing
    missing = network.retrieve_required_inputs([2,8], {1: None})
    assert [] == missing

    # V structure.
    network = ImportanceNetwork(build_cgpms_v_structure())
    missing = network.retrieve_required_inputs([1], {})
    assert [] == missing
    missing = network.retrieve_required_inputs([1, 2], {})
    assert [] == missing
    missing = network.retrieve_required_inputs([8], {1:None, 2:None})
    assert [] == missing
    missing = network.retrieve_required_inputs([2, 8], {1:None})
    assert [] == missing
    missing = network.retrieve_required_inputs([1], {8:None})
    assert [2] == missing
    missing = network.retrieve_required_inputs([2], {8:None})
    assert [1] == missing
    missing = network.retrieve_required_inputs([8], {})
    assert set([1,2]) == set(missing)

    # Markov chain.
    network = ImportanceNetwork(build_cgpms_markov_chain())
    missing = network.retrieve_required_inputs([1,2,8], {})
    assert missing == []
    missing = network.retrieve_required_inputs([2,8], {5:None})
    assert missing == []
    missing = network.retrieve_required_inputs([2], {})
    assert missing == []
    missing = network.retrieve_required_inputs([8], {})
    assert missing == [2]
    missing = network.retrieve_required_inputs([8], {2:None})
    assert missing == []
    missing = network.retrieve_required_inputs([1], {})
    assert set(missing) == set([2,8])
    missing = network.retrieve_required_inputs([1,8], {2:None})
    assert missing == []
    missing = network.retrieve_required_inputs([1], {2:None})
    assert missing == [8]
    missing = network.retrieve_required_inputs([1,2], {})
    assert missing == [8]
    missing = network.retrieve_required_inputs([1,8], {})
    assert missing == [2]
    missing = network.retrieve_required_inputs([1], {8: None})
    assert missing == [2]
    missing = network.retrieve_required_inputs([1,2], {8:None})
    assert missing == []

    # Complex.
    network = ImportanceNetwork(build_cgpms_complex())
    missing = network.retrieve_required_inputs([5,4,3,2], {})
    assert missing == []
    missing = network.retrieve_required_inputs([5,4,3], {2: None})
    assert missing == []
    missing = network.retrieve_required_inputs([3,2], {4:None, 5:None})
    assert missing == []
    missing = network.retrieve_required_inputs([2, 15], {4:None, 5:None})
    assert missing == []
    missing = network.retrieve_required_inputs([2], {4:None})
    assert missing == [5]
    missing = network.retrieve_required_inputs([3, 4], {})
    assert missing == [5]
    missing = network.retrieve_required_inputs([2], {})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_required_inputs([3], {2:None})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_required_inputs([15], {2:None})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_required_inputs([2, 3], {})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_required_inputs([15], {14: None})
    assert set(missing) == set([4, 5])


def test_validate_cgpms():
    c0 = CGpm(outputs=[0,1,5], inputs=[2])
    c1 = CGpm(outputs=[2], inputs=[])
    c2 = CGpm(outputs=[-1], inputs=[5])
    c3 = CGpm(outputs=[1,2], inputs=[7])
    c4 = CGpm(outputs=[], inputs=[])

    # No duplicates.
    helpers.validate_cgpms([c0, c1, c2])
    helpers.validate_cgpms([c2, c3])
    helpers.validate_cgpms([c0])

    # c4 missing output.
    with pytest.raises(ValueError):
        helpers.validate_cgpms([c4])
    with pytest.raises(ValueError):
        helpers.validate_cgpms([c1, c2, c4])

    # c2 collides with c1 and c0
    with pytest.raises(ValueError):
        helpers.validate_cgpms([c0, c3])
    with pytest.raises(ValueError):
        helpers.validate_cgpms([c1, c3])
    with pytest.raises(ValueError):
        helpers.validate_cgpms([c0, c1, c2, c3])


def test_retrieve_ancestors():
    retrieve_ancestors = helpers.retrieve_ancestors

    # No connections.
    cgpms = build_cgpm_no_connection()
    assert set(retrieve_ancestors(cgpms, 2)) == set([])
    assert set(retrieve_ancestors(cgpms, 1)) == set([])
    assert set(retrieve_ancestors(cgpms, 8)) == set([])

    # V structure.
    cgpms = build_cgpms_v_structure()
    assert set(retrieve_ancestors(cgpms, 2)) == set([])
    assert set(retrieve_ancestors(cgpms, 1)) == set([])
    assert set(retrieve_ancestors(cgpms, 8)) == set([1,2])

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    assert set(retrieve_ancestors(cgpms, 2)) == set([])
    assert set(retrieve_ancestors(cgpms, 1)) == set([8,2,5])
    assert set(retrieve_ancestors(cgpms, 8)) == set([2,5])
    with pytest.raises(ValueError):
        retrieve_ancestors(cgpms, 5)

    # Complex.
    cgpms = build_cgpms_complex()
    assert set(retrieve_ancestors(cgpms, 2)) == set([-8,4,-12,5,0,-10,-11])
    assert set(retrieve_ancestors(cgpms, 14)) == set([-8,4,-12,5,0,-10,-11])
    assert set(retrieve_ancestors(cgpms, 3)) == set([-9,4,-12,5,0,-10,-11])
    assert set(retrieve_ancestors(cgpms, 15)) == set([-9,4,-12,5,0,-10,-11])
    assert set(retrieve_ancestors(cgpms, 4)) == set([-12,5,0,-10,-11])
    assert set(retrieve_ancestors(cgpms, 16)) == set([-12,5,0,-10,-11])
    assert set(retrieve_ancestors(cgpms, 5)) == set([0,-10,-11])
    with pytest.raises(ValueError):
        retrieve_ancestors(cgpms, 0)

    # Two parents
    cgpms = build_cgpms_fork()
    assert set(retrieve_ancestors(cgpms, 0)) == set([1,2,3,4])


def test_retrieve_descends():
    retrieve_descendents = helpers.retrieve_descendents

    # No connections.
    cgpms = build_cgpm_no_connection()
    assert set(retrieve_descendents(cgpms, 2)) == set([])
    assert set(retrieve_descendents(cgpms, 1)) == set([])
    assert set(retrieve_descendents(cgpms, 8)) == set([])

    # V structure.
    cgpms = build_cgpms_v_structure()
    assert set(retrieve_descendents(cgpms, 2)) == set([8])
    assert set(retrieve_descendents(cgpms, 1)) == set([8])
    assert set(retrieve_descendents(cgpms, 8)) == set([])

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    assert set(retrieve_descendents(cgpms, 2)) == set([8,1])
    assert set(retrieve_descendents(cgpms, 1)) == set([])
    assert set(retrieve_descendents(cgpms, 8)) == set([1])
    with pytest.raises(ValueError):
        retrieve_descendents(cgpms, 5)

    # Complex.
    cgpms = build_cgpms_complex()
    assert set(retrieve_descendents(cgpms, 2)) == set([])
    assert set(retrieve_descendents(cgpms, 14)) == set([])
    assert set(retrieve_descendents(cgpms, 3)) == set([])
    assert set(retrieve_descendents(cgpms, 15)) == set([])
    assert set(retrieve_descendents(cgpms, 4)) == set([2,14,3,15])
    assert set(retrieve_descendents(cgpms, 16)) == set([])
    assert set(retrieve_descendents(cgpms, 5)) == set([4,16,2,14,3,15])
    with pytest.raises(ValueError):
        retrieve_descendents(cgpms, 0)

    # Two parents
    cgpms = build_cgpms_fork()
    assert set(retrieve_descendents(cgpms, 1)) == set([0])
    assert set(retrieve_descendents(cgpms, 2)) == set([0])
    assert set(retrieve_descendents(cgpms, 0)) == set([])
