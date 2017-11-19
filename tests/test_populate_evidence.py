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

from cgpm.mixtures.view import View
from cgpm.crosscat.state import State

"""Test suite for View._populate_constraints.

Ensures that View._populate_constraints correctly retrieves values from the
dataset.
"""


# ------------------------------------------------------------------------------
# Tests for cgpm.mixtures.view.View

def retrieve_view():
    X = np.asarray([
        [1,    np.nan,        2,      -1,      np.nan],
        [1,         3,        2,      -1,          -5],
        [1,    np.nan,   np.nan,  np.nan,      np.nan],
    ])
    outputs = [0,1,2,3,4]
    return View(
        {c: X[:,c].tolist() for c in outputs},
        outputs=[-1] + outputs,
        cctypes=['normal']*5,
        Zr=[0,1,2]
    )


def test_view_hypothetical_unchanged():
    view = retrieve_view()

    rowid = -1
    targets1 = {3:-1}
    constraints1 = {1:1, 2:2}
    constraints2 = view._populate_constraints(rowid, targets1, constraints1)
    assert constraints1 == constraints2


def test_view_only_rowid_to_populate():
    view = retrieve_view()

    # Can targets X[2,0] for simulate.
    rowid = 2
    targets1 = [0]
    constraints1 = {}
    constraints2 = view._populate_constraints(rowid, targets1, constraints1)
    assert constraints2 == {-1: view.Zr(rowid)}


def test_view_constrain_cluster():
    view = retrieve_view()

    # Cannot constrain cluster assignment of observed rowid.
    rowid = 1
    targets1 = {-1: 2}
    constraints1 = {}
    with pytest.raises(ValueError):
        view._populate_constraints(rowid, targets1, constraints1)


def test_view_values_to_populate():
    view = retrieve_view()

    rowid = 0
    targets1 = [1]
    constraints1 = {4:2}
    constraints2 = view._populate_constraints(rowid, targets1, constraints1)
    assert constraints2 == {0:1, 2:2, 3:-1, 4:2, -1: view.Zr(rowid)}

    rowid = 0
    targets1 = {1:1}
    constraints1 = {4:2}
    constraints2 = view._populate_constraints(rowid, targets1, constraints1)
    assert constraints2 == {2:2, 0:1, 3:-1, 4:2, -1: view.Zr(rowid)}


# ------------------------------------------------------------------------------
# Tests for cgpm.crosscat.state.State

def retrieve_state():
    X = np.asarray([
        [1,    np.nan,        2,      -1,      np.nan],
        [1,         3,        2,      -1,          -5],
        [1,    np.nan,   np.nan,  np.nan,      np.nan],
    ])
    outputs = [0,1,2,3,4]
    return State(
        X,
        outputs=outputs,
        cctypes=['normal']*5,
        Zv={0:0, 1:0, 2:0, 3:0, 4:0},
        Zrv={0:[0,1,2]}
    )

def test_state_constrain_logpdf():
    state = retrieve_state()
    # Cannot targets X[2,0] for logpdf.
    rowid = 2
    targets1 = {0:2}
    constraints1 = {}
    with pytest.raises(ValueError):
        state._validate_cgpm_query(rowid, targets1, constraints1)

def test_state_constrain_errors():
    state = retrieve_state()

    rowid = 1
    targets1 = {1:1, 4:1}
    constraints1 = {}
    with pytest.raises(ValueError):
        state._validate_cgpm_query(rowid, targets1, constraints1)

    rowid = 1
    targets1 = {1:3}
    constraints1 = {4:-5}
    with pytest.raises(ValueError):
        state._validate_cgpm_query(rowid, targets1, constraints1)

    rowid = 1
    targets1 = {0:1, 1:3}
    constraints1 = {}
    with pytest.raises(ValueError):
        state._validate_cgpm_query(rowid, targets1, constraints1)
