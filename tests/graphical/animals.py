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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu
from cgpm.utils import plots as pu
from cgpm.utils import test as tu
from cgpm.utils import render as ru


animals = pd.read_csv('resources/animals/animals.csv', index_col=0)
animal_values = animals.values
animal_names = animals.index.values
animal_features = animals.columns.values


# XXX This function should be parametrized better!
def launch_analysis():
    engine = Engine(
        animals.values.astype(float),
        num_states=64,
        cctypes=['categorical']*len(animals.values[0]),
        distargs=[{'k':2}]*len(animals.values[0]),
        rng=gu.gen_rng(7))

    engine.transition(N=900)
    with open('resources/animals/animals.engine', 'w') as f:
        engine.to_pickle(f)

    engine = Engine.from_pickle(open('resources/animals/animals.engine','r'))
    D = engine.dependence_probability_pairwise()
    pu.plot_clustermap(D)


def render_states_to_disk(filepath, prefix):
    engine = Engine.from_pickle(filepath)
    for i in range(engine.num_states()):
        print '\r%d' % (i,)
        savefile = '%s-%d' % (prefix, i)
        state = engine.get_state(i)
        ru.viz_state(
            state, row_names=animal_names,
            col_names=animal_features, savefile=savefile)


def compare_dependence_heatmap():
    e1 = Engine.from_pickle('resources/animals/animals.engine')
    e2 = Engine.from_pickle('resources/animals/animals-lovecat.engine')

    D1 = e1.dependence_probability_pairwise()
    D2 = e2.dependence_probability_pairwise()
    C1 = pu.plot_clustermap(D1)

    ordering = C1.dendrogram_row.reordered_ind

    fig, ax = plt.subplots(nrows=1, ncols=2)
    pu.plot_heatmap(D1, xordering=ordering, yordering=ordering, ax=ax[0])
    pu.plot_heatmap(D2, xordering=ordering, yordering=ordering, ax=ax[1])
