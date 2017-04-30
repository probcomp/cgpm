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

"""This test suite ensures that simulate and logpdf with zero-density evidence
raises a ValueError."""

import pytest

from cgpm.crosscat.engine import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


@pytest.fixture(scope='module')
def state():
    # Set up the data generation
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'lognormal',
        'beta',
        'vonmises'])
    T, Zv, Zc = tu.gen_data_table(
        30, [1], [[.25, .25, .5]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(0))
    T = T.T
    s = State(
        T,
        cctypes=cctypes,
        distargs=distargs,
        Zv={i: 0 for i in xrange(len(cctypes))},
        rng=gu.gen_rng(0)
    )
    return s


def test_impossible_simulate_evidence(state):
    with pytest.raises(ValueError):
        # Variable 2 is binary-valued, so .8 is impossible.
        state.simulate(-1, [0,1], {2:.8})
    with pytest.raises(ValueError):
        # Variable 3 is lognormal so -1 impossible.
        state.simulate(-1, [4], {3:-1})
    with pytest.raises(ValueError):
        # Variable 4 is beta so 1.1 impossible.
        state.simulate(-1, [5], {3:-1})


def test_impossible_logpdf_evidence(state):
    with pytest.raises(ValueError):
        # Variable 2 is binary-valued, so .8 is impossible.
        state.logpdf(-1, {0:-1}, {2:.8})
    with pytest.raises(ValueError):
        # Variable 3 is lognormal so -1 impossible.
        state.logpdf(-1, {1:1}, {3:-1})
    with pytest.raises(ValueError):
        # Variable 4 is beta so 1.1 impossible.
        state.logpdf(-1, {4:1.1}, {3:-1})


def test_valid_logpdf_query(state):
    # Zero density logpdf is fine.
    state.logpdf(-1, {2:.8}, {1:1.})
    state.logpdf(-1, {5:18})

