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

from cgpm.crosscat.state import State
from cgpm.mixtures import relevance
from cgpm.mixtures.view import View
from cgpm.utils import general as gu
from cgpm.utils import test as tu

def get_data_separated():
    outputs = [1, 2, 3]
    data = np.asarray([
        [0, 0.50, -8.90],
        [0, 0.20, -7.10],
        [1, 1.30, -4.10],
        [1, 1.20, -4.50],
        [1, 1.15, -4.70],
        [0, 1.10, -5.10],
        [2, 2.60, -1.40],
        [1, 2.80, -1.70],
        [3, 8.90, -0.10]
    ])
    assignments = [0, 0, 2, 2, 2, 2, 6, 6, 7]
    return outputs, data, assignments

def view_cgpm_separated():
    outputs, data, assignments = get_data_separated()
    view = View(
        outputs=[1000]+outputs,
        X={output: data[:, i] for i, output in enumerate(outputs)},
        alpha=1.5,
        cctypes=['categorical', 'normal', 'normal'],
        distargs=[{'k': 4}, None, None],
        Zr=assignments,
        rng=gu.gen_rng(1)
    )

    for i in xrange(10):
        view.transition_dim_hypers()
    return view

def state_cgpm_separated():
    outputs, data, assignments = get_data_separated()
    state = State(
        outputs=outputs,
        X=data,
        cctypes=['categorical', 'normal', 'normal'],
        distargs=[{'k':4}, None, None],
        Zv={output: 0 for output in outputs},
        Zrv={0: assignments},
        view_alphas={0: 1.5},
        rng=gu.gen_rng(1)
    )

    for i in xrange(10):
        state.transition_dim_hypers()
    return state

def test_separated():
    """Crash test for relevance_probability."""
    view = view_cgpm_separated()
    # XXX TODO Expand the tests; compuate pairwise, cross-cluster, and
    # multi-cluster relevances.
    rp_view_0 = view.relevance_probability(1, [4, 6, 7], 1)
    rp_view_1 = view.relevance_probability(3, [8], 1)

    assert 0 < np.exp(rp_view_0) < 1  # 0.108687
    assert 0 < np.exp(rp_view_1) < 1  # 0.000366

    state = state_cgpm_separated()
    # Take an enormous leap of faith that, given both State and View are
    # initialized with the same entropy and all values of hyperparameters are
    # sampled in the same order, the relevance probabilities will agree exactly.
    # This is really a test about entropy control.
    rp_state_0 = state.relevance_probability(1, [4, 6, 7], 1)
    rp_state_1 = state.relevance_probability(3, [8], 1)

    assert np.allclose(rp_state_0, rp_view_0)
    assert np.allclose(rp_state_1, rp_view_1)


def test_hypothetical_no_mutation():
    """Ensure using hypothetical rows does not modify state."""
    outputs, data, assignments = get_data_separated()
    state = state_cgpm_separated()

    for i in xrange(10):
        state.transition_dim_hypers()

    # Run a query with two hypothetical rows.
    start_rows = state.n_rows()
    start_marginal = state.logpdf_score()
    rp_state_0 = state.relevance_probability(
        rowid_target=3,
        rowid_query=[8],
        col=1,
        hypotheticals=[{1:1}, {1:2, 3:1}]
    )
    assert state.n_rows() == start_rows
    assert np.allclose(start_marginal, state.logpdf_score())
    assert 0 < np.exp(rp_state_0) < 1


def test_misc_errors():
    # XXX TODO: Add the following test cases:
    # - Unknown rowid_target.
    # - Unknown entry in rowid_query.
    # - Unknown column in relevance probability invocation.
    # - Unknown column in a hypothetical row.
    pass


def test_get_tables_different():
    assert relevance.get_tables_different([0,1]) == (
        [0,1,2], [[1,2], [0,2], [0,1,3]]
    )
    assert relevance.get_tables_different([1,2]) == (
        [1,2,3], [[2,3], [1,3], [1,2,4]]
    )
    assert relevance.get_tables_different([0,4,7,9]) == (
        [0,4,7,9,10],
        [[4,7,9,10], [0,7,9,10], [0,4,9,10], [0,4,7,10], [0,4,7,9,11]]
    )

def test_relevance_commutative_single_query_row():
    """Confirm commutativity of rp for single row queries"""
    view = view_cgpm_separated()
    rp_view_0 = view.relevance_probability(3, [8], 1)
    rp_view_1 = view.relevance_probability(8, [3], 1)
    assert np.allclose(rp_view_0, rp_view_1)

    state = state_cgpm_separated()
    rp_state_0 = state.relevance_probability(3, [8], 1)
    rp_state_1 = state.relevance_probability(8, [3], 1)
    assert np.allclose(rp_state_0, rp_state_1)

def test_relevance_large_concentration_hypers():
    """Confirm crp_alpha -> infty, implies rp(target, query) -> 0."""
    view = view_cgpm_separated()
    lim_view = tu.change_concentration_hyperparameters(view, 1e5)
    rp_view_0 = np.exp(lim_view.relevance_probability(1, [4, 6, 7], 1))
    rp_view_1 = np.exp(lim_view.relevance_probability(3, [8], 1))

    assert np.allclose(rp_view_0, 0, atol=1e-5)
    assert np.allclose(rp_view_1, 0, atol=1e-5)

    state = state_cgpm_separated()
    ext_state = tu.change_concentration_hyperparameters(state, 1e5)
    rp_state_0 = np.exp(ext_state.relevance_probability(1, [4, 6, 7], 1))
    rp_state_1 = np.exp(ext_state.relevance_probability(3, [8], 1))

    assert np.allclose(rp_state_0, 0, atol=1e-5)
    assert np.allclose(rp_state_1, 0, atol=1e-5)

def test_relevance_large_column_hypers():
    """Confirm col_hypers -> infty, implies
    rp(target_1, query_1) == rp(target_2, query_2),
    if query_1 and query_2 are of the same size and
    if all targets and query belong to the same cluster.
    """
    view = view_cgpm_separated()
    lim_view = tu.change_column_hyperparameters(view, 1e5)

    assert not np.allclose(
        view.relevance_probability(2, [3], 1),
        view.relevance_probability(4, [5], 1))

    assert np.allclose(
        lim_view.relevance_probability(2, [3], 1),
        lim_view.relevance_probability(4, [5], 1))

    assert not np.allclose(
        view.relevance_probability(2, [3, 5], 1),
        view.relevance_probability(4, [5, 2], 1))

    assert np.allclose(
        lim_view.relevance_probability(2, [3, 5], 1),
        lim_view.relevance_probability(4, [5, 2], 1))
