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

from cgpm.utils import general as gu
from cgpm.mixtures.view import View

@pytest.fixture
def priorCGPM():
    data = np.random.choice([0, 1], size=(100, 5))
    outputs = range(5)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    model = View(
        X,
        cctypes=['bernoulli']*5,
        outputs=[1000] + outputs,
        rng=gu.gen_rng(0))
    return model

def test_crash_incorporate_missing_value(priorCGPM):
    view = priorCGPM

    # incorporate full row and sample cluster (works)
    normal_row = {i: 1 for i in range(5)}
    view.incorporate(42000, query=normal_row)
    view.unincorporate(42000)

    # incorporate full row and force cluster (works)
    forced_row = {i: 1 for i in range(5) + [view.outputs[0]]}
    view.incorporate(42001, query=forced_row)
    view.unincorporate(42001)

    # incorporate row with missing values (fails)
    missing_row = {i: 1 for i in range(3)}
    view.incorporate(42002, query=missing_row)
    view.unincorporate(42002)

def test_logpdf_missing_value(priorCGPM):
    raise NotImplementedError

def test_logpdf_multirow_missing_values(priorCGPM):
    raise NotImplementedError

def test_simulate_missing_values():
    raise NotImplementedError
