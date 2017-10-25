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
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('categorical'), 'bernoulli')
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('categorical'), 'categorical',
        distargs={'k':2})


def test_poisson_categorical():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('categorical'), 'poisson')
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('categorical'), 'categorical',
        distargs={'k':2})


def test_vonmises_normal():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('vonmises'), 'normal')
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('vonmises'), 'vonmises')

    # Incompatible numeric conversion.
    with pytest.raises(Exception):
        state.update_cctype(CCTYPES.index('normal'), 'vonmises')


def test_geometric_exponential():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=1, progress=False)
    state.update_cctype(CCTYPES.index('geometric'), 'exponential')
    state.transition(N=1, progress=False)

    # Incompatible numeric conversion.
    with pytest.raises(Exception):
        state.update_cctype(CCTYPES.index('exponential'), 'geometric')


def test_categorical_forest():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(1))
    state.transition(N=1, progress=False)
    cat_id = CCTYPES.index('categorical')

    # If cat_id is singleton migrate first.
    if len(state.view_for(cat_id).dims) == 1:
        distargs = DISTARGS[cat_id].copy()
        state.unincorporate_dim(cat_id)
        state.incorporate_dim(
            T[:,cat_id], outputs=[cat_id], cctype='categorical',
            distargs=distargs, v=0)
    state.update_cctype(cat_id, 'random_forest', distargs=distargs)

    bernoulli_id = CCTYPES.index('bernoulli')
    state.incorporate_dim(
        T[:,bernoulli_id], outputs=[191], cctype='bernoulli',
        v=state.Zv(cat_id))
    state.update_cctype(191, 'random_forest', distargs={'k':2})

    # Run valid transitions.
    state.transition(
        N=2, kernels=['rows','column_params','column_hypers'],
        views=[state.Zv(cat_id)], progress=False)

    # Running column transition should raise.
    with pytest.raises(ValueError):
        state.transition(N=1, kernels=['columns'], progress=False)

    # Updating cctype in singleton View should raise.
    distargs = DISTARGS[cat_id].copy()
    state.incorporate_dim(
        T[:,CCTYPES.index('categorical')], outputs=[98],
        cctype='categorical', distargs=distargs, v=max(state.views)+1)
    with pytest.raises(Exception):
        state.update_cctype(98, 'random_forest', distargs=distargs)


def test_categorical_forest_manual_inputs_errors():
    state = State(
        T, cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(1))
    state.transition(N=1, progress=False)
    cat_id = CCTYPES.index('categorical')

    # Put 1201 into the first view.
    view_idx = min(state.views)
    state.incorporate_dim(
        T[:,CCTYPES.index('categorical')], outputs=[1201],
        cctype='categorical', distargs=DISTARGS[cat_id], v=view_idx)

    # Updating cctype with completely invalid input should raise.
    with pytest.raises(Exception):
        distargs = DISTARGS[cat_id].copy()
        distargs['inputs'] = [10000]
        state.update_cctype(1201, 'random_forest', distargs=distargs)

    # Updating cctype with input dimensions outside the view should raise.
    cols_in_view = state.views[view_idx].dims.keys()
    cols_out_view = [c for c in state.outputs if c not in cols_in_view]
    assert len(cols_in_view) > 0 and len(cols_out_view) > 0
    with pytest.raises(Exception):
        distargs = DISTARGS[cat_id].copy()
        distargs['inputs'] = cols_out_view
        state.update_cctype(1201, 'random_forest', distargs=distargs)

    # Updating cctype with no input dimensions should raise.
    with pytest.raises(Exception):
        distargs = DISTARGS[cat_id].copy()
        distargs['inputs'] = []
        state.update_cctype(1201, 'random_forest', distargs=distargs)


def test_linreg_missing_data_ignore():
    dataset = [
        [1, 3, 1],
        [2, 4, 1.5],
        [float('nan'), 5, 1]]
    state = State(dataset, cctypes=['normal']*3, Zv={0:0, 1:0, 2:0},
        rng=gu.gen_rng(1))
    # Make sure that missing covariates are handles as missing cell.
    state.update_cctype(2, 'linear_regression', distargs={'inputs': [0,1]})
    assert state.dim_for(2).inputs[1:] == [0,1]
    state.transition(N=5, kernels=['rows', 'column_hypers', 'view_alphas'])
    state.update_cctype(2, 'normal', distargs={'inputs': [0,1]})
    # Make sure that specified inputs are set correctly.
    state.update_cctype(2, 'linear_regression', distargs={'inputs': [1]})
    assert state.dim_for(2).inputs[1:] == [1]
