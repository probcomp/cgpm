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

import numpy as np
import pytest

from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import general as gu
from cgpm.utils import test as tu

# Set up the data generation
cctypes, distargs = cu.parse_distargs(
    ['normal',
    'poisson',
    'bernoulli',
    'categorical(k=4)',
    'lognormal',
    'exponential',
    'beta',
    'geometric',
    'vonmises'])

T, Zv, Zc = tu.gen_data_table(
    200, [1], [[.25, .25, .5]], cctypes, distargs,
    [.95]*len(cctypes), rng=gu.gen_rng(10))

state = State(T.T, cctypes=cctypes, distargs=distargs, rng=gu.gen_rng(312))
state.transition(N=10, progress=1)

def test_crash_simulate_joint(state):
    state.simulate(-1, [0, 1, 2, 3, 4, 5, 6, 7, 8], N=10)

def test_crash_logpdf_joint(state):
    state.logpdf(-1, {0:1, 1:2, 2:1, 3:3, 4:1, 5:10, 6:.4, 7:2, 8:1.8})

def test_crash_simulate_conditional(state):
    state.simulate(-1, [1, 4, 5, 6, 7, 8], {0:1, 2:1, 3:3}, None, 10)

def test_crash_logpdf_conditional(state):
    state.logpdf(
        -1, {1:2, 4:1, 5:10, 6:.4, 7:2, 8:1.8}, {0:1, 2:1, 3:3})

def test_crash_simulate_joint_observed(state):
    state.simulate(1, [0, 1, 2, 3, 4, 5, 6, 7, 8], None, None, 10)

def test_crash_logpdf_joint_observed(state):
    with pytest.raises(ValueError):
        state.logpdf(1, {0:1, 1:2, 2:1, 3:3, 4:1, 5:10, 6:.4, 7:2, 8:1.8})

def test_crash_simulate_conditional_observed(state):
    with pytest.raises(ValueError):
        state.simulate(1, [1, 4, 5, 6, 7, 8], {0:1, 2:1, 3:3}, None, 10)

def test_crash_logpdf_conditional_observed(state):
    with pytest.raises(ValueError):
        state.logpdf(
            1, {1:2, 4:1, 5:10, 6:.4, 7:2, 8:1.8}, {0:1, 2:1, 3:3})

# Plot!
state.plot()

# Run some solid checks on a complex state.
test_crash_simulate_joint(state)
test_crash_logpdf_joint(state)
test_crash_simulate_conditional(state)
test_crash_logpdf_conditional(state)
test_crash_simulate_joint_observed(state)
test_crash_logpdf_joint_observed(state)
test_crash_simulate_conditional_observed(state)
test_crash_logpdf_conditional_observed(state)

# Joint equals chain rule for state 1.
joint = state.logpdf(-1, {0:1, 1:2})
chain = state.logpdf(-1, {0:1}, {1:2}) + state.logpdf(-1, {1:2})
assert np.allclose(joint, chain)

if False:
    state2 = State(T.T, cctypes=cctypes, distargs=distargs, rng=gu.gen_rng(12))
    state2.transition(N=10, progress=1)

    # Joint equals chain rule for state 2.
    state2.logpdf(-1, {0:1, 1:2})
    state2.logpdf(-1, {0:1}, {1:2}) + state2.logpdf(-1, {1:2})

    # Take the Monte Carlo average of the conditional.
    mc_conditional = np.log(.5) + gu.logsumexp([
        state.logpdf(-1, {0:1}, {1:2}),
        state2.logpdf(-1, {0:1}, {1:2})
    ])

    # Take the Monte Carlo average of the joint.
    mc_joint = np.log(.5) + gu.logsumexp([
        state.logpdf(-1, {0:1, 1:2}),
        state2.logpdf(-1, {0:1, 1:2})
    ])

    # Take the Monte Carlo average of the marginal.
    mc_marginal = np.log(.5) + gu.logsumexp([
        state.logpdf(-1, {1:2}),
        state2.logpdf(-1, {1:2})
    ])
