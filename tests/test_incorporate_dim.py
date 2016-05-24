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

from gpmcc.state import State
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


CCTYPES, DISTARGS = cu.parse_distargs([
    'normal',        # 0
    'poisson',       # 1
    'bernoulli',     # 2
    'lognormal',     # 3
    'exponential',   # 4
    'geometric',     # 5
    'vonmises'])     # 6


T, Zv, Zc = tu.gen_data_table(
    200, [1], [[.33, .33, .34]], CCTYPES, DISTARGS,
    [.95]*len(CCTYPES), rng=gu.gen_rng(0))
T = T.T


def test_incorporate():
    state = State(
        T[:,:2], cctypes=CCTYPES[:2], distargs=DISTARGS[:2], rng=gu.gen_rng(0))
    state.transition(N=5)

    # Incorporate a new dim into view[0].
    state.incorporate_dim(
        T[:,2], outputs=[2], cctype=CCTYPES[2], distargs=DISTARGS[2], v=0)
    assert state.Zv[2] == 0

    # Incorporate a new dim into view[0] with a non-continuous output.
    state.incorporate_dim(
        T[:,2], outputs=[10], cctype=CCTYPES[2], distargs=DISTARGS[2], v=0)
    assert state.Zv[10] == 0
    state.transition(N=10)

    # Incorporate a new dim into view[0] with a non-continiguous .
    # state.incorporate_dim(
    #     T[:,2], outputs=[2], cctype=CCTYPES[2], distargs=DISTARGS[2], v=0)
    # assert state.Zv[2] == 0

    # # Incorporate a new dim into a newly created singleton view.
    # state.incorporate_dim(
    #     T[:,3], outputs=[3], cctype=CCTYPES[3],
    #     distargs=DISTARGS[3], v=len(state.views))
    # assert state.Zv[3] == len(state.views)-1

    # # Incorporate dim without specifying a view.
    # state.incorporate_dim(T[:,4], CCTYPES[4], DISTARGS[4])

    # # Unincorporate first dim.
    # previous = len(state.Zv)
    # state.unincorporate_dim(0)
    # assert len(state.Zv) == previous-1

    # # Reincorporate dim without specifying a view.
    # state.incorporate_dim(T[:,0], CCTYPES[0], DISTARGS[0])

    # # Incorporate dim into singleton view, remove it, assert destroyed.
    # state.incorporate_dim(T[:,5], CCTYPES[5],DISTARGS[5], v=len(state.views))
    # previous = len(state.views)
    # state.unincorporate_dim(5)
    # assert len(state.views) == previous-1

    # # Reincorporate dim into a singleton view.
    # state.incorporate_dim(T[:,5], CCTYPES[4], DISTARGS[4], v=len(state.views))

    # # Incorporate the rest of the dims in the default way.
    # for i in xrange(6, len(CCTYPES)):
    #     state.incorporate_dim(T[:,i], CCTYPES[i], DISTARGS[i])
    # assert state.n_cols() == T.shape[1]

    # # Unincorporate all the dims, except the last one.
    # for i in xrange(state.n_cols()-1, 0, -1):
    #     state.unincorporate_dim(i)
    # assert state.n_cols() == 1

    # # Unincorporating last dim should raise.
    # with pytest.raises(ValueError):
    #     state.unincorporate_dim(0)
