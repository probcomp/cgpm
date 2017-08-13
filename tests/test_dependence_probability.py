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
from cgpm.dummy.barebones import BareBonesCGpm
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


def compute_depprob(d):
    if isinstance(d, list):
        return np.mean(d)
    elif isinstance(d, float):
        return d
    else:
        raise ValueError('Unknown data type for depprob: %s.' % (d,))


def test_dependence_probability():
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'lognormal',
        'beta',
        'vonmises'])

    T, Zv, Zc = tu.gen_data_table(
        100, [.5, .5], [[.25, .25, .5], [.3,.7]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(100))

    T = T.T
    outputs = range(0, 12, 2)

    # Test for direct dependence for state and engine.
    s = State(
        T, outputs=outputs, cctypes=cctypes, distargs=distargs,
        Zv={o:z for o,z in zip(outputs, Zv)}, rng=gu.gen_rng(0))

    e = Engine(
        T, outputs=outputs, cctypes=cctypes, distargs=distargs,
        Zv={o:z for o,z in zip(outputs, Zv)}, rng=gu.gen_rng(0))

    for C in [s,e]:
        for col0, col1 in itertools.product(outputs, outputs):
            i0 = outputs.index(col0)
            i1 = outputs.index(col1)
            assert (
                compute_depprob(C.dependence_probability(col0, col1))
                == (Zv[i0] == Zv[i1])
            )

    # Hook some cgpms into state.

    # XXX What if Zv has only one unique value? Hopefully not with this rng!
    uniques = list(set(Zv))
    parent_1 = [o for i, o in enumerate(outputs) if Zv[i] == uniques[0]]
    parent_2 = [o for i, o in enumerate(outputs) if Zv[i] == uniques[1]]

    c1 = BareBonesCGpm(outputs=[1821, 154], inputs=[parent_1[0]])
    c2 = BareBonesCGpm(outputs=[1721], inputs=[parent_2[0]])
    c3 = BareBonesCGpm(outputs=[9721], inputs=[parent_2[1]])
    c4 = BareBonesCGpm(outputs=[74], inputs=[9721])

    for i, C in enumerate([s, e]):
        C.compose_cgpm(c1 if i==0 else [c1])
        C.compose_cgpm(c2 if i==0 else [c2])
        C.compose_cgpm(c3 if i==0 else [c3])
        C.compose_cgpm(c4 if i==0 else [c4])

        # Between hooked cgpms and state parents.
        for p in parent_1:
            assert compute_depprob(C.dependence_probability(1821, p)) == 1
            assert compute_depprob(C.dependence_probability(154, p)) == 1
            assert compute_depprob(C.dependence_probability(1721, p)) == 0
            assert compute_depprob(C.dependence_probability(9721, p)) == 0
            assert compute_depprob(C.dependence_probability(74, p)) == 0
        for p in parent_2:
            assert compute_depprob(C.dependence_probability(1821, p)) == 0
            assert compute_depprob(C.dependence_probability(154, p)) == 0
            assert compute_depprob(C.dependence_probability(1721, p)) == 1
            assert compute_depprob(C.dependence_probability(9721, p)) == 1
            assert compute_depprob(C.dependence_probability(74, p)) == 1

        # Between hooked cgpm.
        assert compute_depprob(C.dependence_probability(9721, 1721)) == 1
        assert compute_depprob(C.dependence_probability(1821, 154)) == 1
        assert compute_depprob(C.dependence_probability(74, 9721)) == 1
        assert compute_depprob(C.dependence_probability(74, 1721)) == 1

        assert compute_depprob(C.dependence_probability(1821, 1721)) == 0
        assert compute_depprob(C.dependence_probability(1821, 74)) == 0
        assert compute_depprob(C.dependence_probability(154, 74)) == 0


def test_dependence_probability_pairwise():
    cctypes, distargs = cu.parse_distargs(['normal', 'normal', 'normal'])

    T, Zv, _Zc = tu.gen_data_table(
        10, [.5, .5], [[.25, .25, .5], [.3,.7]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(100))

    outputs = [0,1,2]
    engine = Engine(
        T.T, outputs=outputs, cctypes=cctypes, num_states=4,
        distargs=distargs, Zv={o:z for o,z in zip(outputs, Zv)},
        rng=gu.gen_rng(0))

    Ds = engine.dependence_probability_pairwise(multiprocess=0)
    assert len(Ds) == engine.num_states()
    assert all(np.shape(D) == (len(outputs), len(outputs)) for D in Ds)
    for D in Ds:
        for col0, col1 in itertools.product(outputs, outputs):
            i0 = outputs.index(col0)
            i1 = outputs.index(col1)
            actual = D[i0,i1]
            expected = Zv[i0] == Zv[i1]
            assert actual == expected

    Ds = engine.dependence_probability_pairwise(colnos=[0,2], multiprocess=0)
    assert len(Ds) == engine.num_states()
    assert all(np.shape(D) == (2,2) for D in Ds)
