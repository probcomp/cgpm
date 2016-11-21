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

from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu


DATA_NUM_0 = 100
DATA_NUM_1 = 200
NUM_SIM = 10000
NUM_ITER = 5


def test_bernoulli():
    # Switch for multiprocess (0 is faster).
    multiprocess = 0

    # Create categorical data of DATA_NUM_0 zeros and DATA_NUM_1 ones.
    data = np.transpose(np.array([[0] * DATA_NUM_0 + [1] * DATA_NUM_1]))

    # Run a single chain for a few iterations.
    engine = Engine(
        data, cctypes=['categorical'], distargs=[{'k': 2}],
        rng=gu.gen_rng(0), multiprocess=0)
    engine.transition(NUM_ITER, multiprocess=multiprocess)

    # Simulate from hypothetical row and compute the proportion of ones.
    sample = engine.simulate(-1, [0], N=NUM_SIM, multiprocess=multiprocess)[0]
    sum_b = sum(s[0] for s in sample)
    observed_prob_of_1 = (float(sum_b) / float(NUM_SIM))
    true_prob_of_1 = float(DATA_NUM_1) / float(DATA_NUM_0 + DATA_NUM_1)
    # Check 1% relative match.
    assert np.allclose(true_prob_of_1, observed_prob_of_1, rtol=.1)

    # Simulate from observed row as a crash test.
    sample = engine.simulate(1, [0], N=1, multiprocess=multiprocess)

    # Ensure normalized unobserved probabilities.
    p0_uob = engine.logpdf(-1, {0:0}, multiprocess=multiprocess)[0]
    p1_uob = engine.logpdf(-1, {0:1}, multiprocess=multiprocess)[0]
    assert np.allclose(gu.logsumexp([p0_uob, p1_uob]), 0)

    # A logpdf query constraining an observed returns an error.
    with pytest.raises(ValueError):
        engine.logpdf(1, {0:0}, multiprocess=multiprocess)
    with pytest.raises(ValueError):
        engine.logpdf(1, {0:1}, multiprocess=multiprocess)
