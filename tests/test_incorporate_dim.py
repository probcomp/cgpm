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
    10, [1], [[.33, .33, .34]], CCTYPES, DISTARGS,
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
    state.transition(N=1)

    # Incorporate a new dim into view[0] with a non-continuous output.
    state.incorporate_dim(
        T[:,2], outputs=[10], cctype=CCTYPES[2], distargs=DISTARGS[2], v=0)
    assert state.Zv[10] == 0
    state.transition(N=1)

    # Some crash testing queries.
    state.logpdf(-1, [(10,1)], evidence=[(0,2), (1,1)])
    state.simulate(-1, [10], evidence=[(0,2)])

    # Incorporating with a duplicated output should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[10], cctype=CCTYPES[2], distargs=DISTARGS[2], v=0)

    # Multivariate incorporate should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[10, 2], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=0)

    # Missing output should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=0)

    # Wrong number of rows should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2][:-1], outputs=[11], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=0)

    # Inputs should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[11], inputs=[2], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=0)

    # Incorporate dim into a newly created singleton view.
    state.incorporate_dim(
        T[:,3], outputs=[3], cctype=CCTYPES[3],
        distargs=DISTARGS[3], v=state.n_views())
    assert state.Zv[3] == state.n_views()-1
    state.transition(N=1)

    # Incorporate dim without specifying a view.
    state.incorporate_dim(T[:,4], outputs=[4],
        cctype=CCTYPES[4], distargs=DISTARGS[4])
    state.transition(N=1)

    # Unincorporate first dim.
    previous = state.n_cols()
    state.unincorporate_dim(0)
    assert state.n_cols() == previous-1
    state.transition(N=1)

    # Reincorporate dim without specifying a view.
    state.incorporate_dim(
        T[:,0], outputs=[0], cctype=CCTYPES[0], distargs=DISTARGS[0])
    state.transition(N=1)

    # Incorporate dim into singleton view, remove it, assert destroyed.
    state.incorporate_dim(
        T[:,5], outputs=[5], cctype=CCTYPES[5], distargs=DISTARGS[5],
        v=state.n_views())
    previous = state.n_views()
    state.unincorporate_dim(5)
    assert state.n_views() == previous-1
    state.transition(N=1)

    # Reincorporate dim into a singleton view.
    state.incorporate_dim(T[:,5], outputs=[5], cctype=CCTYPES[5],
        distargs=DISTARGS[5], v=len(state.views))
    state.transition(N=1)

    # Incorporate the rest of the dims in the default way.
    for i in xrange(6, len(CCTYPES)):
        state.incorporate_dim(
            T[:,i], outputs=[max(state.outputs)+1],
            cctype=CCTYPES[i], distargs=DISTARGS[i])
    state.transition(N=1)

    # Unincorporating non-existent dim should raise.
    with pytest.raises(ValueError):
        state.unincorporate_dim(9999)

    # Unincorporate all the dims, except the last one.
    for o in state.outputs[:-1]:
        state.unincorporate_dim(o)
    assert state.n_cols() == 1
    state.transition(N=1)

    # Unincorporating last dim should raise.
    with pytest.raises(ValueError):
        state.unincorporate_dim(state.outputs[0])
