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

from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu
from cgpm.utils import plots as pu
from cgpm.utils import test as tu


D, Zv, Zc = tu.gen_data_table(
    n_rows=150,
    view_weights=None,
    cluster_weights=[[.2,.2,.2,.4], [.3,.2,.5],],
    cctypes=['normal']*6,
    distargs=[None]*6,
    separation=[0.95]*6,
    view_partition=[0,0,0,1,1,1],
    rng=gu.gen_rng(12))


@pytest.mark.parametrize('lovecat', [True, False])
def test_two_views_row_partition__ci_(lovecat):
    engine = Engine(
        D.T, cctypes=['normal']*len(D),
        Zv={0:0, 1:0, 2:0, 3:1, 4:1, 5:1},
        rng=gu.gen_rng(12), num_states=64)
    if lovecat:
        engine.transition_lovecat(
            N=100,
            kernels=[
                'row_partition_hyperparameters'
                'row_partition_assignments',
                'column_hyperparameters',
        ])
    else:
        engine.transition(
            N=100,
            kernels=[
                'view_alphas',
                'rows',
                'column_hypers',
        ])

    R1 = engine.row_similarity_pairwise(cols=[0,1,2])
    R2 = engine.row_similarity_pairwise(cols=[3,4,5])

    # XXX TODO: Find a way to test the actual row similarity matrices with the
    # theoretical Zc structure.


@pytest.mark.parametrize('lovecat', [True, False])
def test_two_views_column_partition__ci_(lovecat):
    engine = Engine(
        D.T, cctypes=['normal']*len(D), rng=gu.gen_rng(12), num_states=64)

    if lovecat:
        engine.transition_lovecat(N=200)
    else:
        engine.transition(N=200, multiprocess=1)

    P = engine.dependence_probability_pairwise()
    R1 = engine.row_similarity_pairwise(cols=[0,1,2])
    R2 = engine.row_similarity_pairwise(cols=[3,4,5])

    pu.plot_clustermap(P)
    pu.plot_clustermap(R1)
    pu.plot_clustermap(R2)

    P_THEORY = [
        [1,1,1,0,0,0],
        [1,1,1,0,0,0],
        [1,1,1,0,0,0],
        [0,0,0,1,1,1],
        [0,0,0,1,1,1],
        [0,0,0,1,1,1],
    ]

    # XXX TODO: Find a way to test the actual dependence probability matrix
    # with the THEORY matrix.
