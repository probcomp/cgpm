# -*- coding: utf-8 -*-

# The MIT License (MIT)

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

from gpmcc.utils import test as tu
from gpmcc.utils import config as cu
from gpmcc import state
import numpy as np

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

from gpmcc import engine
runner = engine.Engine(T.T, cctypes, distargs, num_states=8, initialize=0)
runner.initialize(multithread=0)
runner.transition(N=1000, multithread=True)
