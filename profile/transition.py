#!/usr/bin/env python

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project.
# Released under the MIT License; refer to LICENSE.txt.


import os
os.environ['GPMCCDEBUG'] = '0'

import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import test as tu

rng = np.random.RandomState(10)

# Set up the data generation
cctypes, distargs = cu.parse_distargs(['normal', 'normal', 'normal', 'normal',])

T, Zv, Zc = tu.gen_data_table(
    n_rows=200,
    view_weights=[.5,.5],
    cluster_weights=[[.25, .25, .5], [.5, .5]],
    cctypes=cctypes,
    distargs=distargs,
    separation=[.95]*len(cctypes),
    rng=rng)

state = State(T.T, cctypes=cctypes, distargs=distargs, rng=rng)
state.transition(N=10, progress=0)
