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

from cgpm.crosscat.state import State
from cgpm.utils import general as gu


X = [[1,     np.nan,     2,         -1,         np.nan  ],
     [1,     3,          2,         -1,         -5      ],
     [18,    -7,         -2,        11,         -12     ],
     [1,     np.nan,     np.nan,    np.nan,     np.nan  ],
     [18,    -7,         -2,        11,         -12     ]]


def get_state():
    return State(
        X,
        outputs=range(5),
        cctypes=['normal']*5,
        Zv={0:0, 1:0, 2:0, 3:1, 4:1},
        rng=gu.gen_rng(0),
    )


def test_invalid_evidence_keys():
    state = get_state()
    # Non-existent view -3.
    with pytest.raises(ValueError):
        state.incorporate(
            state.n_rows(),
            {0:0, 1:1, 2:2, 3:3, 4:4, state.crp_id_view+2:0}
        )


def test_invalid_evidence():
    state = get_state()
    # Evidence is disabled since State has no inputs.
    with pytest.raises(Exception):
        state.incorporate(state.n_rows(), {0:0, 1:1, 2:2, 3:3, 4:4}, {12:1})


def test_invalid_cluster():
    state = get_state()
    # Should crash with None.
    with pytest.raises(Exception):
        state.incorporate(
            state.n_rows(),
            {0:0, 1:1, 2:2, 3:3, 4:4, state.views[0].outputs[0]:None})


def test_invalid_query_nan():
    state = get_state()
    # Not allowed to incorporate nan.
    with pytest.raises(ValueError):
        state.incorporate(state.n_rows(), {0:np.nan, 1:1, 2:2, 3:3, 4:4})


def test_invalid_rowid():
    state = get_state()
    # Non-contiguous rowids disabled.
    for rowid in range(state.n_rows()):
        with pytest.raises(ValueError):
            state.incorporate(rowid, {0:2})

def test_incorporate_valid():
    state = get_state()
    # Incorporate row into cluster 0 for all views.
    previous = np.asarray([state.views[v].Nk(0) for v in [0,1]])
    state.incorporate(
        state.n_rows(),
        {0:0, 1:1, 2:2, 3:3, 4:4, state.views[0].outputs[0]:0,
            state.views[1].outputs[0]:0}
    )
    assert [state.views[v].Nk(0) for v in [0,1]] == list(previous+1)
    # Incorporate row into cluster 0 for view 1 with some missing values.
    previous = state.views[1].Nk(0)
    state.incorporate(state.n_rows(), {0:0, 2:2, state.views[1].outputs[0]:0})
    assert state.views[1].Nk(0) == previous+1
    state.transition(N=2)
    # Hypothetical cluster 100.
    view = state.views[state.views.keys()[0]]
    state.incorporate(
        state.n_rows(),
        {0:0, 1:1, 2:2, 3:3, 4:4, view.outputs[0]:100})


def test_unincorporate():
    state = get_state()
    # Unincorporate all the rows except for the last one.
    # XXX Must remove the last rowid only at each invocation.
    rowids = range(0, state.n_rows())
    for rowid in rowids[:-1]:
        with pytest.raises(ValueError):
            state.unincorporate(rowid)
    # Remove rowids starting from state.n_rows()-1 down to 1.
    for rowid in reversed(rowids[1:]):
        state.unincorporate(rowid)
    assert state.n_rows() == 1
    # Cannot unincorporate the final rowid.
    with pytest.raises(ValueError):
        state.unincorporate(0)
    state.transition(N=2)


def test_incorporate_session():
    rng = gu.gen_rng(4)
    state = State(
        X, cctypes=['normal']*5, Zv={0:0, 1:0, 2:1, 3:1, 4:2}, rng=rng)
    # Incorporate row into a singleton cluster for all views.
    previous = [len(state.views[v].Nk()) for v in [0,1,2]]
    data = {i: rng.normal() for i in xrange(5)}
    clusters = {
        state.views[0].outputs[0]: previous[0],
        state.views[1].outputs[0]: previous[1],
        state.views[2].outputs[0]: previous[2],
    }
    state.incorporate(state.n_rows(), gu.merged(data, clusters))
    assert [len(state.views[v].Nk()) for v in [0,1,2]] == \
        [p+1 for p in previous]
    # Incorporate row without specifying clusters, and some missing values
    data = {i: rng.normal() for i in xrange(2)}
    state.incorporate(state.n_rows(), data)
    state.transition(N=3)
    # Remove the incorporated rowid.
    state.unincorporate(state.n_rows()-1)
    state.transition(N=3)
