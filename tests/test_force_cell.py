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


def test_invalid_nonnan_cell():
    state = get_state()
    # rowid 0 and output 1 is not nan.
    with pytest.raises(ValueError):
        state.force_cell(0, {0: 1})

def test_invalid_variable():
    state = get_state()
    # Output variable 10 does not exist.
    with pytest.raises(KeyError):
        state.force_cell(0, {10: 1})

def test_invalid_rowid():
    state = get_state()
    # Cannot force non-incorporated rowid.
    with pytest.raises(ValueError):
        state.force_cell(10, {0: 1})
    with pytest.raises(ValueError):
        state.force_cell(None, {0: 1})

def test_force_cell_valid():
    state = get_state()
    # Retrieve normal component model to force cell (0,1)
    rowid, dim = 0, 1
    view = state.view_for(dim)
    k = view.Zr(rowid)
    normal_component = state.dim_for(dim).clusters[k]
    # Initial sufficient statistics.
    N_initial = normal_component.N
    sum_x_initial = normal_component.sum_x
    sum_x_sq_initial = normal_component.sum_x_sq
    # Force!
    state.force_cell(0, {1: 1.5})
    # Confirm incremented statistics.
    assert normal_component.N == N_initial + 1
    assert np.allclose(normal_component.sum_x, sum_x_initial + 1.5)
    assert np.allclose(normal_component.sum_x_sq, sum_x_sq_initial + 1.5**2)
    # Cannot force again.
    with pytest.raises(ValueError):
        state.force_cell(0, {1: 1})
    # Run a transition.
    state.transition(N=1)
    # Force cell (3,[1,2])
    state.force_cell(3, {1: -7, 2:-2})
    state.transition(N=1)
