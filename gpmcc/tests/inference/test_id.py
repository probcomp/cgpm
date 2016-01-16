# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gpmcc.engine import Engine
from gpmcc.utils import config as cu

np.random.seed(0)

N_ROWS = 300
N_STATES = 12
N_ITERS = 50

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
engine = Engine(X, cctypes, distargs, num_states=N_STATES, initialize=True)
engine.initialize()
engine.transition(N=N_ITERS)

# Dependence probability.
D = engine.dependence_probability_pairwise()
zmat = sns.clustermap(D, yticklabels=column_names, xticklabels=column_names)
plt.setp(zmat.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(zmat.ax_heatmap.get_xticklabels(), rotation=90)
plt.show()
