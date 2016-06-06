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

import pytest

from gpmcc.network import helpers
from gpmcc.network.importance import ImportanceNetwork


CGpm = namedtuple('CGpm', ['outputs', 'inputs'])


def build_cgpm_no_connection():
    return [
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[1], inputs=[]),
        CGpm(outputs=[8], inputs=[])]


def build_cgpms_v_structure():
    return [
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[1], inputs=[]),
        CGpm(outputs=[8], inputs=[2, 1])]


def build_cgpms_markov_chain():
    return [
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[1], inputs=[8]),
        CGpm(outputs=[8], inputs=[2, 5]),]


def build_cgpms_complex():
    return [
        CGpm(outputs=[2, 14], inputs=[4, 5, -8]),
        CGpm(outputs=[3, 15], inputs=[4, -9]),
        CGpm(outputs=[5], inputs=[0, -10, -11]),
        CGpm(outputs=[4, 16], inputs=[5, -12]),]



def test_retrieve_variable_to_cgpm():
    cgpms = [
        CGpm(outputs=[0, 1, 5], inputs=[2]),
        CGpm(outputs=[2], inputs=[]),
        CGpm(outputs=[-1], inputs=[7]),]
    for order in itertools.permutations(cgpms):
        variable_to_cgpm = helpers.retrieve_variable_to_cgpm(order)
        for v, c in variable_to_cgpm.iteritems():
            assert v in order[c].outputs

def test_retrieve_adjacency():
    # No connections.
    cgpms = build_cgpm_no_connection()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency(cgpms, vtc)
    assert {0: [], 1:[], 2:[]}, set([]) == adj

    # V structure.
    cgpms = build_cgpms_v_structure()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency(cgpms, vtc)
    assert {0: [], 1:[], 2:[0, 1]} == adj

    # Markov chain.
    cgpms = build_cgpms_markov_chain()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency(cgpms, vtc)
    assert {0: [], 1:[2], 2:[0]} == adj

    # Complex.
    cgpms = build_cgpms_complex()
    vtc = helpers.retrieve_variable_to_cgpm(cgpms)
    adj = helpers.retrieve_adjacency(cgpms, vtc)
    assert {0: [2,3], 1:[3], 2:[], 3:[2]} == adj


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


def test_retrieve_missing_inputs():
    # No connections.
    network = ImportanceNetwork(build_cgpm_no_connection())
    missing = network.retrieve_missing_inputs([2], {8: None})
    assert [] == missing
    missing = network.retrieve_missing_inputs([2,8], {1: None})
    assert [] == missing

    # V structure.
    network = ImportanceNetwork(build_cgpms_v_structure())
    missing = network.retrieve_missing_inputs([1], {})
    assert [] == missing
    missing = network.retrieve_missing_inputs([1, 2], {})
    assert [] == missing
    missing = network.retrieve_missing_inputs([8], {1:None, 2:None})
    assert [] == missing
    missing = network.retrieve_missing_inputs([2, 8], {1:None})
    assert [] == missing
    missing = network.retrieve_missing_inputs([1], {8:None})
    assert [2] == missing
    missing = network.retrieve_missing_inputs([2], {8:None})
    assert [1] == missing
    missing = network.retrieve_missing_inputs([8], {})
    assert set([1,2]) == set(missing)

    # Markov chain.
    network = ImportanceNetwork(build_cgpms_markov_chain())
    missing = network.retrieve_missing_inputs([1,2,8], {})
    assert missing == []
    missing = network.retrieve_missing_inputs([2,8], {5:None})
    assert missing == []
    missing = network.retrieve_missing_inputs([2], {})
    assert missing == []
    missing = network.retrieve_missing_inputs([8], {})
    assert missing == [2]
    missing = network.retrieve_missing_inputs([8], {2:None})
    assert missing == []
    missing = network.retrieve_missing_inputs([1], {})
    assert set(missing) == set([2,8])
    missing = network.retrieve_missing_inputs([1,8], {2:None})
    assert missing == []
    missing = network.retrieve_missing_inputs([1], {2:None})
    assert missing == [8]
    missing = network.retrieve_missing_inputs([1,2], {})
    assert missing == [8]
    missing = network.retrieve_missing_inputs([1,8], {})
    assert missing == [2]
    missing = network.retrieve_missing_inputs([1], {8: None})
    assert missing == [2]
    missing = network.retrieve_missing_inputs([1,2], {8:None})
    assert missing == []

    # Complex.
    network = ImportanceNetwork(build_cgpms_complex())
    missing = network.retrieve_missing_inputs([5,4,3,2], {})
    assert missing == []
    missing = network.retrieve_missing_inputs([5,4,3], {2: None})
    assert missing == []
    missing = network.retrieve_missing_inputs([3,2], {4:None, 5:None})
    assert missing == []
    missing = network.retrieve_missing_inputs([2, 15], {4:None, 5:None})
    assert missing == []
    missing = network.retrieve_missing_inputs([2], {4:None})
    assert missing == [5]
    missing = network.retrieve_missing_inputs([3, 4], {})
    assert missing == [5]
    missing = network.retrieve_missing_inputs([2], {})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_missing_inputs([3], {2:None})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_missing_inputs([15], {2:None})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_missing_inputs([2, 3], {})
    assert set(missing) == set([4, 5])
    missing = network.retrieve_missing_inputs([15], {14: None})
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
