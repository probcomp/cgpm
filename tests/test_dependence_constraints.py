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

from __future__ import absolute_import
from builtins import range
import itertools
import pytest

import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils import general as gu
from cgpm.utils import validation as vu


def test_naive_bayes_independence():
    rng = gu.gen_rng(1)
    D = rng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = list(itertools.combinations(list(range(10)), 2))
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
    outputs = list(range(10, 20))
    with pytest.raises(ValueError):
        State(T, outputs=list(range(10,20)), cctypes=['normal']*10,
            Cd=[(2,0)], rng=rng)
    with pytest.raises(ValueError):
        State(T, outputs=list(range(10,20)), cctypes=['normal']*10,
            Ci=[(2,0)], rng=gu.gen_rng(0))
