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

import pytest

import numpy as np

from gpmcc.crosscat.state import State


@pytest.fixture(scope='module')
def state():
    X = [[1, np.nan, 2, -1, np.nan],
        [1, 3, 2, -1, -5],
        [1, np.nan, np.nan, np.nan, np.nan]]
    s = State(X, cctypes=['normal']*5)
    return s


def test_hypothetical_unchanged(state):
    rowid = -1
    qr1 = [(3,-1)]
    ev1 = [(1,1), (2,2)]
    ev2 = state._populate_evidence(rowid, qr1, ev1)
    assert set(ev1) == set(ev2)


def test_nothing_to_populate(state):
    rowid = 2
    qr1 = [(0,1)]
    ev2 = state._populate_evidence(rowid, qr1, [])
    assert set(ev2) == set([])


def test_some_to_populate(state):
    rowid = 0
    qr1 = [(1,1)]
    ev1 = [(2,2)]
    ev2 = state._populate_evidence(rowid, qr1, ev1)
    assert set(ev2) == set([(2,2), (0,1), (3,-1)])


def test_everything_to_populate(state):
    rowid = 1
    qr1 = [(1,1), (4,1)]
    ev2 = state._populate_evidence(rowid, qr1, [])
    assert set(ev2) == set([(0,1), (2,2), (3,-1)])

    qr1 = [(1,3)] # Actual query not allowed.
    ev1 = [(4,-5)]
    ev2 = state._populate_evidence(rowid, qr1, ev1)
    assert set(ev2) == set([(0,1), (2,2), (3,-1), (4,-5)])

    qr1 = [(0,1), (1,3)] # Actual query not allowed.
    ev1 = [(2,2)]
    ev2 = state._populate_evidence(rowid, qr1, [])
    assert set(ev2) == set([(2,2), (3,-1), (4,-5)])
