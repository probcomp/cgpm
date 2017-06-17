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

"""Test suite targeting cgpm.crosscat.engine.add_state."""

import pytest

from cgpm.crosscat.engine import Engine
from cgpm.dummy.twoway import TwoWay
from cgpm.utils import general as gu


def get_engine():
    X = [[0.123, 1, 0], [1.12, 0, 1], [1.1, 1, 2]]
    rng = gu.gen_rng(1)
    return Engine(
        X,
        outputs=[8,7,9],
        num_states=4,
        cctypes=['normal', 'bernoulli', 'categorical'],
        distargs=[None, None, {'k': 3}],
        rng=rng
    )


def test_engine_add_state_basic():
    engine = get_engine()
    initial_num_states = engine.num_states()
    engine.add_state()
    assert engine.num_states() == initial_num_states + 1
    engine.add_state(count=2)
    assert engine.num_states() == initial_num_states + 3
    engine.transition(N=3)
    engine.drop_state(6)
    assert engine.num_states() == initial_num_states + 2


def test_engine_add_state_custom():
    # Add a state with a specified view and row partition.
    engine = get_engine()
    engine.add_state(count=2, Zv={7:0, 8:1, 9:0}, Zrv={0: [0,1,1], 1: [1,1,1]})
    new_state = engine.get_state(engine.num_states()-1)
    assert new_state.Zv() == {7:0, 8:1, 9:0}
    assert new_state.views[0].Zr(0) == 0
    assert new_state.views[0].Zr(1) == 1
    assert new_state.views[0].Zr(2) == 1
    assert new_state.views[1].Zr(0) == 1
    assert new_state.views[1].Zr(1) == 1
    assert new_state.views[1].Zr(2) == 1


def test_engine_add_state_kwarg_errors():
    engine = get_engine()
    with pytest.raises(ValueError):
        # Cannot specify new dataset.
        engine.add_state(X=[[0,1]])
    with pytest.raises(ValueError):
        # Cannot specify new outputs.
        engine.add_state(outputs=[1,2])
    with pytest.raises(ValueError):
        # Cannot specify new cctypes.
        engine.add_state(cctypes=['normal', 'normal'])
    with pytest.raises(ValueError):
        # Cannot specify new distargs.
        engine.add_state(distargs=[None, None, {'k' : 3}])
    with pytest.raises(ValueError):
        # Cannot specify all together.
        engine.add_state(X=[[0,1]], outputs=[1,2], cctypes=['normal', 'normal'])


def test_engine_add_state_composite_errors():
    # XXX Add a Github ticket to support this feature. User should provide all
    # the composite cgpms to match the count of initialized models.
    engine = get_engine()
    engine.compose_cgpm([
        TwoWay(outputs=[4], inputs=[7]) for _i in xrange(engine.num_states())
    ])
    with pytest.raises(ValueError):
        engine.add_state()
