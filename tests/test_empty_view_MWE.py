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

from cgpm.utils import general as gu
from cgpm.mixtures.view import View

def test_create_prior_view():  # works
    data = np.random.choice([0, 1], size=(100, 5))
    outputs = range(5)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    View(
        X,
        cctypes=['bernoulli']*5,
        outputs=[1000] + outputs,
        rng=gu.gen_rng(0))

def test_create_empty_view():  # fails
    outputs = range(5)
    X = {c: [] for c in outputs}
    View(
        X,
        cctypes=['bernoulli']*5,
        outputs=[1000] + outputs,
        rng=gu.gen_rng(0))
