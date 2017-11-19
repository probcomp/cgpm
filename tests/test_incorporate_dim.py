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

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


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


def test_incorporate_engine():
    engine = Engine(
        T[:,:2],
        cctypes=CCTYPES[:2],
        distargs=DISTARGS[:2],
        num_states=4,
        rng=gu.gen_rng(0),
    )
    engine.transition(N=5)

    # Incorporate a new dim into with a non-contiguous output.
    engine.incorporate_dim(
        T[:,2],
        outputs=[10],
        cctype=CCTYPES[2],
        distargs=DISTARGS[2]
    )
    engine.transition(N=2)

    # Serialize the engine, and run a targeted transtion on variable 10.
    m = engine.to_metadata()
    engine2 = Engine.from_metadata(m)
    engine2.transition(N=2, cols=[10], multiprocess=0)
    assert all(s.outputs == [0,1,10] for s in engine.states)

def test_incorporate_state():
    state = State(
        T[:,:2], cctypes=CCTYPES[:2], distargs=DISTARGS[:2], rng=gu.gen_rng(0))
    state.transition(N=5)

    target = state.views.keys()[0]

    # Incorporate a new dim into view[0].
    state.incorporate_dim(
        T[:,2], outputs=[2], cctype=CCTYPES[2], distargs=DISTARGS[2], v=target)
    assert state.Zv(2) == target
    state.transition(N=1)

    # Incorporate a new dim into view[0] with a non-contiguous output.
    state.incorporate_dim(
        T[:,2], outputs=[10], cctype=CCTYPES[2], distargs=DISTARGS[2], v=target)
    assert state.Zv(10) == target
    state.transition(N=1)

    # Some crash testing queries.
    state.logpdf(-1, {10:1}, constraints={0:2, 1:1})
    state.simulate(-1, [10], constraints={0:2})

    # Incorporating with a duplicated output should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[10], cctype=CCTYPES[2], distargs=DISTARGS[2],
            v=target)

    # Multivariate incorporate should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[10, 2], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=target)

    # Missing output should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=target)

    # Wrong number of rows should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2][:-1], outputs=[11], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=target)

    # Inputs should raise.
    with pytest.raises(ValueError):
        state.incorporate_dim(
            T[:,2], outputs=[11], inputs=[2], cctype=CCTYPES[2],
            distargs=DISTARGS[2], v=target)

    # Incorporate dim into a newly created singleton view.
    target = max(state.views)+1
    state.incorporate_dim(
        T[:,3], outputs=[3], cctype=CCTYPES[3],
        distargs=DISTARGS[3], v=target)
    assert state.Zv(3) == target
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
    target = max(state.views)+1
    state.incorporate_dim(
        T[:,5], outputs=[5], cctype=CCTYPES[5], distargs=DISTARGS[5],
        v=target)
    previous = len(state.views)
    state.unincorporate_dim(5)
    assert len(state.views) == previous-1
    state.transition(N=1)

    # Reincorporate dim into a singleton view.
    target = max(state.views)+1
    state.incorporate_dim(T[:,5], outputs=[5], cctype=CCTYPES[5],
        distargs=DISTARGS[5], v=target)
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
