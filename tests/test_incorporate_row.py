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
from gpmcc.utils import general as gu


X = [[1,     np.nan,     2,         -1,         np.nan  ],
     [1,     3,          2,         -1,         -5      ],
     [18,    -7,         -2,        11,         -12     ],
     [1,     np.nan,     np.nan,    np.nan,     np.nan  ],
     [18,    -7,         -2,        11,         -12     ]]


def test_invalid_evidence_keys():
    state = State(X, cctypes=['normal']*5, Zv=[0,0,0,1,1], rng=gu.gen_rng(0))
    # Non-existent view -3.
    with pytest.raises(ValueError):
        state.incorporate(
            rowid=-1,
            query={0:0, 1:1, 2:2, 3:3, 4:4, -1:0, -2:0, -3:0})
    # Condition on an output 1.
    with pytest.raises(ValueError):
        state.incorporate(
            rowid=-1,
            query={0:0, 1:1, 2:2, 3:3, 4:4, -3:0})


def test_invalid_evidence_cluster():
    state = State(X, cctypes=['normal']*5, Zv=[0,0,0,1,1], rng=gu.gen_rng(0))
    # Should crash with None.
    with pytest.raises(Exception):
        state.incorporate(
            rowid=-1,
            query={0:0, 1:1, 2:2, 3:3, 4:4, -1:None})


def test_invalid_query_nan():
    state = State(X, cctypes=['normal']*5, Zv=[0,0,0,1,1], rng=gu.gen_rng(0))
    # Not allowed to incorporate nan.
    with pytest.raises(ValueError):
        state.incorporate(
            rowid=-1,
            query={0:np.nan, 1:1, 2:2, 3:3, 4:4})


def test_invalid_rowid():
    state = State(X, cctypes=['normal']*5, Zv=[0,0,0,1,1], rng=gu.gen_rng(0))
    # Hypotheticals disabled.
    with pytest.raises(ValueError):
        state.incorporate(
            rowid=0,
            query={0:2})


def test_incorporate_valid():
    state = State(X, cctypes=['normal']*5, Zv=[0,0,1,1,1], rng=gu.gen_rng(0))
    # Incorporate row into cluster 0 for all views.
    previous = np.asarray([v.Nk[0] for v in state.views])
    state.incorporate(
        rowid=-1,
        query={0:0, 1:1, 2:2, 3:3, 4:4, -1:0, -2:0})
    assert [v.Nk[0] for v in state.views] == list(previous+1)
    # Incorporate row into cluster 0 for view -2 with some missing values.
    previous = state.views[1].Nk[0]
    state.incorporate(
        rowid=-1,
        query={0:0, 2:2, -2:0})
    assert state.views[1].Nk[0] == previous+1
    state.transition(N=2)
    # Hypothetical cluster 100.
    state.incorporate(
        rowid=-1,
        query={0:0, 1:1, 2:2, 3:3, 4:4, -1:100})


def test_incorporate_session():
    rng = gu.gen_rng(4)
    state = State(X, cctypes=['normal']*5, Zv=[0,0,1,1,2], rng=rng)
    # Incorporate row into a singleton cluster for all views.
    previous = [len(v.Nk) for v in state.views]
    data = {i: rng.normal() for i in xrange(5)}
    clusters = {-1: previous[0], -2: previous[1], -3: previous[2]}
    state.incorporate(-1, gu.merged(data, clusters))
    assert [len(v.Nk) for v in state.views] == [p+1 for p in previous]
    # Incorporate row without specifying clusters, and some missing values
    previous = [len(v.Nk) for v in state.views]
    data = {i: rng.normal() for i in xrange(2)}
    state.incorporate(-1, data)
    state.transition(N=2)
