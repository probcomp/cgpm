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
import numpy as np
import pytest

from collections import namedtuple

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


CGpm = namedtuple('CGpm', ['outputs', 'inputs'])

def test_dependence_probability():
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'lognormal',
        'beta_uc',
        'vonmises'])

    T, Zv, Zc = tu.gen_data_table(
        100, [.5, .5], [[.25, .25, .5], [.3,.7]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(100))

    T = T.T
    # Make some nan cells for evidence.
    outputs = range(0, 12, 2)
    s = State(
        T, outputs=outputs, cctypes=cctypes, distargs=distargs,
        Zv={o:z for o,z in zip(outputs, Zv)}, rng=gu.gen_rng(0))

    # Test for direct dependence.
    for col0, col1 in itertools.product(outputs, outputs):
        i0 = outputs.index(col0)
        i1 = outputs.index(col1)
        assert s.dependence_probability(col0, col1) == (Zv[i0] == Zv[i1])

    # Now hook some cgpms.
    uniques = list(set(Zv))
    parent_1 = [o for i, o in enumerate(outputs) if Zv[i] == uniques[0]]
    parent_2 = [o for i, o in enumerate(outputs) if Zv[i] == uniques[1]]

    c1 = CGpm(outputs=[1821, 154], inputs=[parent_1[0]])
    c2 = CGpm(outputs=[1721], inputs=[parent_2[0]])
    c3 = CGpm(outputs=[9721], inputs=[parent_2[1]])

    s.compose_cgpm(c1)
    s.compose_cgpm(c2)
    s.compose_cgpm(c3)

    assert s.dependence_probability(1821, 1721) == 0
    assert s.dependence_probability(9721, 1721) == 1
    assert s.dependence_probability(1821, 154) == 1
