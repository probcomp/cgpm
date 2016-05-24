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
import unittest

import numpy as np

from gpmcc.state import State
from gpmcc.utils import general as gu
from gpmcc.utils import validation as vu


def test_naive_bayes():
    D = np.random.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = list(itertools.combinations(range(10), 2))
    state = State(T, cctypes=['normal']*10, Ci=Ci, rng=gu.gen_rng(0))
    state.transition(N=10, do_progress=0)
    vu.validate_crp_constrained_partition(state.Zv, [], Ci, {}, {})

def test_complex_relationships():
    D = np.random.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Ci = [(2,8), (0,3)]
    state = State(T, cctypes=['normal']*10, Ci=Ci, rng=gu.gen_rng(0))
    state.transition(N=10, do_progress=0)
    vu.validate_crp_constrained_partition(state.Zv, [], Ci, {}, {})
