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

from cgpm.crosscat.engine import Engine
from cgpm.utils import config as cu
from cgpm.utils import test as tu

# Set up the data generation.
n_rows = 200
view_weights = np.asarray([0.55, .45])
cluster_weights = [np.array([.33, .33, .34]), np.array([.1, .9])]
cctypes = [
    'beta_uc',
    'normal',
    'poisson',
    'categorical(k=4)',
    'vonmises',
    'bernoulli',
    'lognormal',
    'normal',
    'normal']

separation = [.95] * len(cctypes)
cctypes, distargs = cu.parse_distargs(cctypes)
T, Zv, Zc = tu.gen_data_table(n_rows, view_weights, cluster_weights, cctypes,
    distargs, separation)

cctypes = [
    'normal',
    'normal',
    'categorical(k=%d)' % (max(T[2])+1),
    'categorical(k=4)',
    'normal',
    'categorical(k=2)',
    'normal',
    'normal',
    'normal']
cctypes, distargs = cu.parse_distargs(cctypes)

engine = Engine(
    T.T, cctypes=cctypes, distargs=distargs, num_states=8)
engine.transition(N=1000, multithread=True)
