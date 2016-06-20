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
import seaborn as sns

from cgpm.crosscat.engine import Engine
from cgpm.utils import config as cu

np.random.seed(0)

N_ROWS = 300
N_STATES = 12
N_ITERS = 100

cctypes = ['categorical(k={})'.format(N_ROWS)] + ['normal']*8
cctypes, distargs = cu.parse_distargs(cctypes)
column_names = ['id'] + ['one cluster']*4 + ['four cluster']*4

# id column.
X = np.zeros((N_ROWS, 9))
X[:,0] = np.arange(N_ROWS)

# Four columns of one cluster from the standard normal.
X[:,1:5] = np.random.randn(N_ROWS, 4)

# Four columns of four clusters with unit variance and means \in {0,1,2,3}.
Z = np.random.randint(4, size=(N_ROWS))
X[:,5:] = 4*np.reshape(np.repeat(Z,4), (len(Z),4)) + np.random.randn(N_ROWS, 4)

# Inference.
engine = Engine(
    X, cctypes=cctypes, distargs=distargs, num_states=N_STATES)
engine.transition(N=N_ITERS)

# Dependence probability.
D = engine.dependence_probability_pairwise()
zmat = sns.clustermap(D, yticklabels=column_names, xticklabels=column_names)
plt.setp(zmat.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(zmat.ax_heatmap.get_xticklabels(), rotation=90)
plt.show()
