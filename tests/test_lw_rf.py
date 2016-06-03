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

"""Crash and sanity tests for queries using likelihood weighting inference
with a RandomForest component model. Not an inference quality test suite."""

import pytest

import numpy as np

from gpmcc.crosscat.state import State
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


@pytest.fixture(scope='module')
def state():
    cctypes, distargs = cu.parse_distargs([
        'categorical(k=5)',
        'normal',
        'poisson',
        'bernoulli'])
    T, Zv, Zc = tu.gen_data_table(
        50, [1], [[.33, .33, .34]], cctypes, distargs, [.95]*len(cctypes),
        rng=gu.gen_rng(0))
    s = State(
        T.T, cctypes=cctypes, distargs=distargs, Zv=[0]*len(cctypes),
        rng=gu.gen_rng(0))
    s.update_cctype(0, 'random_forest', distargs={'k':5})
    # XXX Uncomment me for a bug!
    # state.update_cctype(1, 'linear_regression')
    s.transition(
        N=10,
        kernels=['rows','view_alphas','alpha','column_params','column_hypers'])
    return s


def test_simulate_unconditional__ci_(state):
    for rowid in [-1, 1]:
        samples = state.simulate(rowid, [0], N=2)
        check_members(samples, range(5))


def test_simulate_conditional__ci_(state):
    samples = state.simulate(
        -1, [0], evidence={1:-1, 2:1, 3:1}, N=2)
    check_members(samples, range(5))
    samples = state.simulate(-1, [0, 2, 3], N=2)
    check_members(samples, range(5))
    samples = state.simulate(1, [0, 2, 3], N=2)
    check_members(samples, range(5))


def test_logpdf_unconditional__ci_(state):
    for rowid, k in zip([-1, 1], xrange(5)):
        assert state.logpdf(rowid, {0: k}) < 0


def test_logpdf_deterministic__ci_(state):
    # Ensure logpdf estimation deterministic when all parents in evidence.
    for k in xrange(5):
        lp1 = state.logpdf(
            -1, {0:k, 3:0}, evidence={1:1, 2:1})
        lp2 = state.logpdf(
            -1, {0:k, 3:0}, evidence={1:1, 2:1})
        assert np.allclose(lp1, lp2)
    # Observed cell already has parents in evidence.
    for k in xrange(5):
        lp1 = state.logpdf(1, {0:k, 3:0})
        lp2 = state.logpdf(1, {0:k, 3:0})
        assert np.allclose(lp1, lp2)


def check_members(samples, allowed):
    for s in samples:
        assert s[0] in allowed
