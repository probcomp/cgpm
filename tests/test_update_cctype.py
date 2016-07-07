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

from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


CCTYPES, DISTARGS = cu.parse_distargs([
    'normal',
    'poisson',
    'categorical(k=2)',
    'bernoulli',
    'lognormal',
    'exponential',
    'geometric',
    'vonmises'])

T, Zv, Zc = tu.gen_data_table(
    20, [1], [[.33, .33, .34]], CCTYPES, DISTARGS,
    [.95]*len(CCTYPES), rng=gu.gen_rng(0))

T = T.T


def test_categorical_bernoulli():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('categorical'), 'bernoulli')
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('categorical'), 'categorical',
        distargs={'k':2})


def test_poisson_categorical():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('categorical'), 'poisson')
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('categorical'), 'categorical',
        distargs={'k':2})


def test_vonmises_normal():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('vonmises'), 'normal')
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('vonmises'), 'vonmises')

    # Incompatible numeric conversion.
    with pytest.raises(Exception):
        state.update_cctype(CCTYPES.index('normal'), 'vonmises')


def test_geometric_exponential():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1)
    state.update_cctype(CCTYPES.index('geometric'), 'exponential')
    state.transition(N=1)

    # Incompatible numeric conversion.
    with pytest.raises(Exception):
        state.update_cctype(CCTYPES.index('exponential'), 'geometric')


def test_categorical_forest():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(1))
    state.transition(N=1)
    cat_id = CCTYPES.index('categorical')
    cat_distargs = DISTARGS[cat_id]

    # If cat_id is singleton migrate first.
    if len(state.view_for(cat_id).dims) == 1:
        state.unincorporate_dim(cat_id)
        state.incorporate_dim(
            T[:,cat_id], outputs=[120], cctype='categorical',
            distargs=cat_distargs, v=0)
        cat_id = 120
    state.update_cctype(cat_id, 'random_forest', distargs=cat_distargs)

    bernoulli_id = CCTYPES.index('bernoulli')
    state.incorporate_dim(
        T[:,bernoulli_id], outputs=[191], cctype='bernoulli',
        v=state.Zv(cat_id))
    state.update_cctype(191, 'random_forest', distargs={'k':2})

    # Run valid transitions.
    state.transition(
        N=2, kernels=['rows','column_params','column_hypers'],
        views=[state.Zv(cat_id)])

    # Running column transition should raise.
    with pytest.raises(ValueError):
        state.transition(N=1, kernels=['columns'])

    # Updating cctype in singleton View should raise.
    state.incorporate_dim(
        T[:,CCTYPES.index('categorical')], outputs=[98],
        cctype='categorical', distargs=cat_distargs, v=len(state.views))
    with pytest.raises(Exception):
        state.update_cctype(98, 'random_forest', distargs=cat_distargs)
