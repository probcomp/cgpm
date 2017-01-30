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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu
from cgpm.utils import plots as pu
from cgpm.utils import test as tu


# -------------------- Normal Component Models ------------------------------- #

def retrieve_normal_dataset():
    D, Zv, Zc = tu.gen_data_table(
        n_rows=150,
        view_weights=None,
        cluster_weights=[[.2,.2,.2,.4], [.3,.2,.5],],
        cctypes=['normal']*6,
        distargs=[None]*6,
        separation=[0.95]*6,
        view_partition=[0,0,0,1,1,1],
        rng=gu.gen_rng(12))
    return D


@pytest.mark.parametrize('lovecat', [True, False])
def test_two_views_row_partition_normal__ci_(lovecat):
    D = retrieve_normal_dataset()

    engine = Engine(
        D.T, cctypes=['normal']*len(D),
        Zv={0:0, 1:0, 2:0, 3:1, 4:1, 5:1},
        rng=gu.gen_rng(12), num_states=64)

    if lovecat:
        engine.transition_lovecat(
            N=100,
            kernels=[
                'row_partition_assignments',
                'row_partition_hyperparameters',
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

    pu.plot_clustermap(R1)
    pu.plot_clustermap(R2)
    return engine


@pytest.mark.parametrize('lovecat', [True, False])
def test_two_views_column_partition_normal__ci_(lovecat):
    D = retrieve_normal_dataset()

    engine = Engine(
        D.T, outputs=[5,0,1,2,3,4],
        cctypes=['normal']*len(D), rng=gu.gen_rng(12), num_states=64)

    if lovecat:
        engine.transition_lovecat(N=200)
    else:
        engine.transition(N=200)

    P = engine.dependence_probability_pairwise()
    R1 = engine.row_similarity_pairwise(cols=[5,0,1])
    R2 = engine.row_similarity_pairwise(cols=[2,3,4])

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
    return engine

# -------------------- Bernoulli Component Models ---------------------------- #

def retrieve_bernoulli_dataset():
    D, Zv, Zc = tu.gen_data_table(
        n_rows=150,
        view_weights=None,
        cluster_weights=[[.5,.5], [.1,.9],],
        cctypes=['bernoulli']*4,
        distargs=[None]*4,
        separation=[0.95]*4,
        view_partition=[0,0,1,1],
        rng=gu.gen_rng(12))
    return D


@pytest.mark.parametrize('lovecat', [True, False])
def test_two_views_row_partition_bernoulli__ci_(lovecat):
    D = retrieve_bernoulli_dataset()

    if lovecat:
        engine = Engine(
            D.T,
            cctypes=['categorical']*len(D),
            distargs=[{'k':2}]*len(D),
            Zv={0:0, 1:0, 2:1, 3:1},
            rng=gu.gen_rng(12), num_states=64)
        engine.transition_lovecat(
            N=100,
            kernels=[
                'row_partition_assignments',
                'row_partition_hyperparameters',
                'column_hyperparameters',
        ])
    else:
        engine = Engine(
            D.T,
            cctypes=['bernoulli']*len(D),
            Zv={0:0, 1:0, 2:1, 3:1},
            rng=gu.gen_rng(12), num_states=64)
        engine.transition(
            N=100,
            kernels=[
                'view_alphas',
                'rows',
                'column_hypers',
        ])

    R1 = engine.row_similarity_pairwise(cols=[0,1])
    R2 = engine.row_similarity_pairwise(cols=[2,3])

    pu.plot_clustermap(R1)
    pu.plot_clustermap(R2)
    return engine


@pytest.mark.parametrize('lovecat', [True, False])
def test_two_views_column_partition_bernoulli__ci_(lovecat):
    D = retrieve_bernoulli_dataset()

    engine = Engine(
        D.T,
        cctypes=['categorical']*len(D),
        distargs=[{'k':2}]*len(D),
        rng=gu.gen_rng(12),
        num_states=64)
    if lovecat:
        engine.transition_lovecat(N=200)
    else:
        # engine = Engine(
        #     D.T,
        #     cctypes=['bernoulli']*len(D),
        #     rng=gu.gen_rng(12),
        #     num_states=64)
        engine.transition(N=200)

    P = engine.dependence_probability_pairwise()
    R1 = engine.row_similarity_pairwise(cols=[0,1])
    R2 = engine.row_similarity_pairwise(cols=[2,3])

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
    return engine
