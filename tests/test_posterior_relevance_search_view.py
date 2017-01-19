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
from itertools import product
from collections import OrderedDict
from pprint import pprint

from cgpm.crosscat.state import State
from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, merged


trivial_data = [[0, 0],
                [1, 0],
                [0, 1],
                [1, 1]]

def initialize_trivial_view():
    data = np.array(trivial_data)
    R = len(data)
    D = len(data[0])
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    Zr = [0 for i in range(R)]
    view = View(
        X,
        outputs=[1000] + outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        hypers={
            i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zr=Zr)
    return view

def load_animals_view():
    with open('tests/resources/animals_state.pkl', 'rb') as f:
        animals_state = State.from_pickle(f)
    view = animals_state.views[65]
    return view

# ----- TEST SEARCH ORDER AS EXPECTED ----- #
def check_search_order_as_expected(view, query, ordered_list):
    search_query_result = view.posterior_relevance_search(query, debug=True)
    search_list = [pair[0] for pair in search_query_result]
    return search_list == ordered_list

def test_order_trivial_nonhypothetical_row_0():
    view = initialize_trivial_view()
    query = {0: {}}
    assert (check_search_order_as_expected(view, query, [1, 2, 3]) or 
            check_search_order_as_expected(view, query, [2, 1, 3]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_nonhypothetical_row_1():
    view = initialize_trivial_view()
    query = {1: {}}
    assert (check_search_order_as_expected(view, query, [0, 3, 2]) or 
            check_search_order_as_expected(view, query, [3, 0, 2]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_nonhypothetical_row_2():
    view = initialize_trivial_view()
    query = {2: {}}
    assert (check_search_order_as_expected(view, query, [0, 3, 1]) or 
            check_search_order_as_expected(view, query, [3, 0, 1]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_nonhypothetical_row_3():
    view = initialize_trivial_view()
    query = {3: {}}
    assert (check_search_order_as_expected(view, query, [1, 2, 0]) or 
            check_search_order_as_expected(view, query, [2, 1, 0]))
    print view.posterior_relevance_search(query)
    

def test_order_trivial_hypothetical_row_0():
    view = initialize_trivial_view()
    query = {-1: {0: 0, 1: 0}}
    assert (check_search_order_as_expected(view, query, [0, 1, 2, 3]) or 
            check_search_order_as_expected(view, query, [0, 2, 1, 3]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_hypothetical_row_1():
    view = initialize_trivial_view()
    query = {-1: {0: 1, 1: 0}}
    assert (check_search_order_as_expected(view, query, [1, 0, 3, 2]) or 
            check_search_order_as_expected(view, query, [1, 3, 0, 2]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_hypothetical_row_2():
    view = initialize_trivial_view()
    query = {-1: {0: 0, 1: 1}}
    assert (check_search_order_as_expected(view, query, [2, 0, 3, 1]) or 
            check_search_order_as_expected(view, query, [2, 3, 0, 1]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_hypothetical_row_3():
    view = initialize_trivial_view()
    query = {-1: {0: 1, 1: 1}}
    assert (check_search_order_as_expected(view, query, [3, 1, 2, 0]) or 
            check_search_order_as_expected(view, query, [3, 2, 1, 0]))
    print view.posterior_relevance_search(query)
    
def test_order_trivial_hypothetical_rows_0_0_1():
    view = initialize_trivial_view()
    query = {4: {0: 0, 1: 0},
             5: {0: 0, 1: 0},
             6: {0: 1, 1: 1}}
    assert check_search_order_as_expected(view, query, [0, 1, 2, 3])
         
# ----- TEST WEAK COMMUTATIVITY ----- #         
def assert_weak_commutativity(view, row_x, row_y):
    first_results = view.posterior_relevance_search({-1: row_x, -2: row_y}, True)
    second_results = view.posterior_relevance_search({-1: row_y, -2: row_x}, True)
    assert np.allclose([pair[1] for pair in first_results],
                       [pair[1] for pair in second_results])
    assert np.allclose([pair[0] for pair in first_results[:2]],
                       [pair[0] for pair in second_results[:2]])

def test_weak_commutativity_trivial_rows_0_1():
    view = initialize_trivial_view()
    row0 = {0: 0, 1: 0}
    row1 = {0: 1, 1: 0}
    assert_weak_commutativity(view, row0, row1)

def test_weak_commutativity_trivial_rows_0_3():
    view = initialize_trivial_view()
    row0 = {0: 0, 1: 0}
    row3 = {0: 1, 1: 1}
    assert_weak_commutativity(view, row0, row3)
    
# ----- TEST COMMUTATIVITY ----- #
# This test might be wrong
# Counter-example: -x1------x2--x3 
# Closest to x1: x2
# Closest to x2: x3
def assert_commutativity(view, query):
    # Target is the top result after searching for query 
    search_query_results = view.posterior_relevance_search(query)
    target_id = search_query_results[0][0]
    target = {target_id: {}}

    # New target is the top result after searching for target
    search_target_results = view.posterior_relevance_search(target)
    new_target_id = search_target_results[0][0]
    new_target = {new_target_id: {}}

    # New target should be the same as query
    assert new_target == query

# ----- DEACTIVATED TESTS ----- #
# def test_relevance_search_wrt_rows_in_first_cluster():
#     view = initialize_trivial_view()

#     for rowid in xrange(4):
#         score = view.posterior_relevance_search(
#             query={rowid: {}}, debug=True)

#         # Assert highest scoring values come from same cluster as query
#         first_four = [score[i][0] for i in xrange(3)]
#         first_cluster = range(4)
#         assert set(first_four) == set(first_cluster)

#         # Lowest scoring values come from different cluster than query?
#         last_four = [score[i][0] for i in xrange(4, 8)]
#         second_cluster = range(4, 8)
#         assert set(last_four) == set(second_cluster)

# def test_relevance_search_wrt_rows_in_second_cluster():
#     view = initialize_trivial_view()

#     for rowid in xrange(4, 8):
#         score = view.posterior_relevance_search(query={rowid: {}}, debug=True)

#         # Assert highest score with itself
#         assert score[0][0] == rowid

#         # Assert highest scoring values come from same cluster as query
#         first_four = [score[i][0] for i in xrange(4)]
#         second_cluster = range(4, 8)
#         assert set(first_four) == set(second_cluster)

#         # Assert lowest scoring values come from different cluster than query
#         last_four = [score[i][0] for i in xrange(4, 8)]
#         first_cluster = range(4)
#         assert set(last_four) == set(first_cluster)

# def test_relevance_search_mixed():
#     view = initialize_trivial_view()

#     score = view.posterior_relevance_search(query={0: {}, 7: {}}, debug=True)

#     # Assert highest scores with itself
#     first_two = [score[i][0] for i in xrange(2)]
#     assert set(first_two) == {0, 7}

#     pprint(score)

# def test_relevance_search_wrt_majority():
#     view = initialize_trivial_view()

#     score = view.posterior_relevance_search(query={0: {}, 1: {}, 7: {}}, debug=True)

#     # Assert highest scoring values come from majority cluster in query
#     first_four = [score[i][0] for i in xrange(4)]
#     first_cluster = range(4)
#     assert set(first_four) == set(first_cluster)

#     # Assert lowest scoring values come from minority cluster in query
#     last_four = [score[i][0] for i in xrange(4, 8)]
#     second_cluster = range(4, 8)
#     assert set(last_four) == set(second_cluster)

#     pprint(score)
