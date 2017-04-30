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
import pytest

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
    cctypes = ['categorical', 'normal', 'normal']
    distargs = [{'k': 4}, None, None]
    return outputs, data, assignments, cctypes, distargs

def get_data_all_ones():
    outputs = [1]
    n_rows = 200
    n_cols = 1
    data = np.ones(shape=[n_rows, n_cols])
    assignments = [0] * n_rows
    cctypes = ['bernoulli']
    distargs = [None]
    return outputs, data, assignments, cctypes, distargs

def get_data_missing():
    outputs = [1]
    n_rows = 199
    n_cols = 1
    data = np.vstack((np.nan, np.ones(shape=[n_rows, n_cols])))
    assignments = [0] * (n_rows+1)
    cctypes = ['bernoulli']
    distargs = [None]
    return outputs, data, assignments, cctypes, distargs

def gen_view_cgpm(get_data):
    outputs, data, assignments, cctypes, distargs = get_data()
    view = View(
        outputs=[1000]+outputs,
        X={output: data[:, i] for i, output in enumerate(outputs)},
        alpha=1.5,
        cctypes=cctypes,
        distargs=distargs,
        Zr=assignments,
        rng=gu.gen_rng(1)
    )

    for i in xrange(10):
        view.transition_dim_hypers()
    return view

def gen_state_cgpm(get_data):
    outputs, data, assignments, cctypes, distargs = get_data()
    state = State(
        outputs=outputs,
        X=data,
        cctypes=cctypes,
        distargs=distargs,
        Zv={output: 0 for output in outputs},
        Zrv={0: assignments},
        view_alphas={0: 1.5},
        rng=gu.gen_rng(1)
    )

    for i in xrange(10):
        state.transition_dim_hypers()
    return state

@pytest.mark.xfail(strict=True, reason='state and view compute different scores.')
def test_separated():
    """Crash test for relevance_probability."""
    view = gen_view_cgpm(get_data_separated)
    # XXX TODO Expand the tests; compute pairwise, cross-cluster, and
    # multi-cluster relevances.
    rp_view_0 = view.relevance_probability(1, [4, 6, 7], 1)
    rp_view_1 = view.relevance_probability(3, [8], 1)

    assert 0 < rp_view_0 < 1  # 0.108687
    assert 0 < rp_view_1 < 1  # 0.000366

    state = gen_state_cgpm(get_data_separated)
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
    state = gen_state_cgpm(get_data_separated)

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
    assert rp_state_0 in [0, 1]


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
    view = gen_view_cgpm(get_data_separated)
    rp_view_0 = view.relevance_probability(3, [8], 1)
    rp_view_1 = view.relevance_probability(8, [3], 1)
    assert np.allclose(rp_view_0, rp_view_1)

    state = gen_state_cgpm(get_data_separated)
    rp_state_0 = state.relevance_probability(3, [8], 1)
    rp_state_1 = state.relevance_probability(8, [3], 1)
    assert np.allclose(rp_state_0, rp_state_1)

def test_relevance_large_concentration_hypers():
    """Confirm crp_alpha -> infty, implies rp(target, query) -> 0."""
    view = gen_view_cgpm(get_data_separated)
    lim_view = tu.change_concentration_hyperparameters(view, 1e5)
    rp_view_0 = lim_view.relevance_probability(1, [4, 6, 7], 1)
    rp_view_1 = lim_view.relevance_probability(3, [8], 1)

    assert np.allclose(rp_view_0, 0, atol=1e-5)
    assert np.allclose(rp_view_1, 0, atol=1e-5)

    state = gen_state_cgpm(get_data_separated)
    ext_state = tu.change_concentration_hyperparameters(state, 1e5)
    rp_state_0 = ext_state.relevance_probability(1, [4, 6, 7], 1)
    rp_state_1 = ext_state.relevance_probability(3, [8], 1)

    assert np.allclose(rp_state_0, 0, atol=1e-5)
    assert np.allclose(rp_state_1, 0, atol=1e-5)

def test_relevance_large_column_hypers():
    """Confirm col_hypers -> infty, implies
    rp(target_1, query_1) == rp(target_2, query_2),
    if query_1 and query_2 are of the same size and
    if all targets and query belong to the same cluster.
    """
    view = gen_view_cgpm(get_data_separated)
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

def test_relevance_with_itself():
    """Confirm that rp(target, target)==1, for any target."""
    state = gen_state_cgpm(get_data_separated)
    assert np.allclose(state.relevance_probability(2, [2], 1), 1.0)

def test_relevance_analytically():
    view = gen_view_cgpm(get_data_all_ones)
    n = view.n_rows()
    a = view.alpha()  # crp_alpha
    b1 = view.dims[1].hypers['alpha']  # bernoulli pseudocounts for one
    b0 = view.dims[1].hypers['beta']  # bernoulli pseudocounts for zero

    # Compute rp(t_rowid=0, q_rowid=[1])
    rp_computational = view.relevance_probability(0, [1], 1)

    # Compute Pr[zT = zQ, xT, xQ, S] =
    #   Pr[t,Q|zT=zQ=0] * Pr[zT=zQ=0] + Pr[t,Q|zT=zQ=1] * Pr[zT=zQ=1]
    #   analytically for t_rowid=0, Q_rowid=[1]
    p_same_table = (
        (n-2+b1)/(b1+b0+n-2) * (n-1+b1)/(b1+b0+n-1) *
        (n-2)/(n-2+a) * (n-1)/(n-1+a) +
        b1/(b1+b0) * (b1+1)/(b1+b0+1) *
        a/(n-2+a) * 1/(n-1+a))

    # Compute Pr[zT \ne zQ, xT, xQ, S] =
    #   Pr[t,Q|zT=0, zQ=1] * Pr[zT=0, zQ=1] +
    #     Pr[t,Q|zT=1, zQ=0] * Pr[zT=1, zQ=0] +
    #     Pr[t,Q|zT=1, zQ=2] * Pr[zT=1, zQ=2]
    #   analytically for t_rowid=0, Q_rowid=[1]
    p_diff_table = (
        (n-2+b1)/(b1+b0+n-2) * b1/(b1+b0) * (n-2)/(n-2+a) * a/(n-1+a) +
        b1/(b1+b0) * (n-2+b1)/(b1+b0+n-2) * a/(n-1+a) * (n-2)/(n-2+a) +
        b1/(b1+b0) * b1/(b1+b0) * a/(n-1+a) * a/(n-2+a))

    # rp(t, Q) = Pr[zT = zQ, xT, xQ, S] /
    #    (Pr[zT = zQ, xT, xQ, S] +  Pr[zT \ne zQ, xT, xQ, S])
    rp_analytical = p_same_table / (p_same_table + p_diff_table)

    assert np.allclose(rp_analytical, rp_computational)

def test_crash_missing():
    view = gen_view_cgpm(get_data_missing)
    rp_view_0 = view.relevance_probability(0, [1], 1)
    rp_view_1 = view.relevance_probability(1, [0], 1)

    assert 0 <= rp_view_0 <= 1.
    assert 0 <= rp_view_1 <= 1.
    assert np.allclose(rp_view_0, rp_view_1)

def test_missing_analytical():
    view = gen_view_cgpm(get_data_missing)
    n = view.n_rows()
    a = view.alpha()  # crp_alpha
    b1 = view.dims[1].hypers['alpha']  # bernoulli pseudocounts for one
    b0 = view.dims[1].hypers['beta']  # bernoulli pseudocounts for zero

    # Compute rp(t_rowid=0, q_rowid=[1])
    rp_computational = view.relevance_probability(0, [1], 1)

    # Compute Pr[zT = zQ, xQ, S] =
    #   Pr[Q|zT=zQ=0] * Pr[zT=zQ=0] + Pr[Q|zT=zQ=1] * Pr[zT=zQ=1]
    #   analytically for t_rowid=0, Q_rowid=[1].
    p_numerator = (
        (n-2+b1)/(b1+b0+n-2) * (n-2)/(n-2+a) * (n-1)/(n-1+a) +
        b1/(b1+b0) * a/(n-2+a) * 1/(n-1+a))

    # Compute Pr[xQ] =
    #   Pr[Q|zQ=0] * Pr[zQ=0] + Pr[Q|zQ=1] * Pr[zQ=1]
    #   analytically for Q_rowid=[1].
    p_denominator = (
        (n-2+b1)/(b1+b0+n-2) * (n-2)/(n-2+a) +
        b1/(b1+b0) * a/(n-2+a))

    rp_analytical = p_numerator / p_denominator

    assert np.allclose(rp_computational, rp_analytical)
