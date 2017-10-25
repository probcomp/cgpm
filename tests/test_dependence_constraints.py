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

import itertools
import pytest

import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils import general as gu
from cgpm.utils import validation as vu

from markers import integration


def test_naive_bayes_independence():
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = list(itertools.combinations(range(10), 2))
    state = State(T, cctypes=['normal']*10, Ci=Ci, rng=rng)
    state.transition(N=10, progress=0)
    vu.validate_crp_constrained_partition(state.Zv(), [], Ci, {}, {})


def test_complex_independent_relationships():
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = [(2,8), (0,3)]
    state = State(T, cctypes=['normal']*10, Ci=Ci, rng=rng)
    state.transition(N=10, progress=0)
    vu.validate_crp_constrained_partition(state.Zv(), [], Ci, {}, {})


CIs = [[], [(2,8), (0,3)]]
@pytest.mark.parametrize('Ci', CIs)
def test_simple_dependence_constraint(Ci):
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Cd = [(2,0), (8,3)]
    state = State(T, cctypes=['normal']*10, Ci=Ci, Cd=Cd, rng=rng)
    with pytest.raises(ValueError):
        # Cannot transition columns with dependencies.
        state.transition(N=10, kernels=['columns'], progress=0)
    state.transition(
        N=10,
        kernels=['rows', 'alpha', 'column_hypers', 'alpha', 'view_alphas'],
        progress=False)
    vu.validate_crp_constrained_partition(state.Zv(), Cd, Ci, {}, {})


def test_zero_based_outputs():
    """Constraints must have zero-based output variables for now."""
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    outputs = range(10, 20)
    with pytest.raises(ValueError):
        State(T, outputs=range(10,20), cctypes=['normal']*10,
            Cd=[(2,0)], rng=rng)
    with pytest.raises(ValueError):
        State(T, outputs=range(10,20), cctypes=['normal']*10,
            Ci=[(2,0)], rng=gu.gen_rng(0))

@integration
def test_naive_bayes_independence_lovecat():
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = list(itertools.combinations(range(10), 2))
    state = State(T, cctypes=['normal']*10, Ci=Ci, rng=gu.gen_rng(0))
    state.transition(N=10, progress=0)
    vu.validate_crp_constrained_partition(state.Zv(), [], Ci, {}, {})
    state.transition_lovecat(N=100, progress=0)
    vu.validate_crp_constrained_partition(state.Zv(), [], Ci, {}, {})


@integration
def test_complex_independent_relationships_lovecat():
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = [(2,8), (0,3)]
    Cd = [(2,3), (0,8)]
    state = State(T, cctypes=['normal']*10, Ci=Ci, Cd=Cd, rng=gu.gen_rng(0))
    state.transition_lovecat(N=1000, progress=1)
    vu.validate_crp_constrained_partition(state.Zv(), Cd, Ci, {}, {})

@integration
def test_independence_inference_quality_lovecat():
    rng = gu.gen_rng(584)
    column_view_1 = rng.normal(loc=0, size=(50,1))

    column_view_2 = np.concatenate((
        rng.normal(loc=10, size=(25,1)),
        rng.normal(loc=20, size=(25,1)),
    ))

    data_view_1 = np.repeat(column_view_1, 4, axis=1)
    data_view_2 = np.repeat(column_view_2, 4, axis=1)
    data = np.column_stack((data_view_1, data_view_2))

    Zv0 = {i: 0 for i in xrange(8)}
    state = State(data, Zv=Zv0, cctypes=['normal']*8, rng=gu.gen_rng(10))
    state.transition_lovecat(N=100, progress=1)
    for col in [0, 1, 2, 3,]:
        assert state.Zv(col) == state.Zv(0)
    for col in [4, 5, 6, 7]:
        assert state.Zv(col) == state.Zv(4)
    assert state.Zv(0) != state.Zv(4)

    # Get lovecat to merge the dependent columns into one view.
    Cd = [(0,1), (2,3), (4,5), (6,7)]
    Zv0 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3}
    state = State(data, Zv=Zv0, cctypes=['normal']*8, Cd=Cd, rng=gu.gen_rng(1))
    state.transition_lovecat(N=100, progress=1)
    for col in [0, 1, 2, 3,]:
        assert state.Zv(col) == state.Zv(0)
    for col in [4, 5, 6, 7]:
        assert state.Zv(col) == state.Zv(4)
    assert state.Zv(0) != state.Zv(4)
