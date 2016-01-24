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

from gpmcc.utils import test as tu
from gpmcc import engine
from gpmcc.state import State
import numpy as np

# This script generates a column of every data type and plots inference
# in real time.

# Set up the data generation.
n_rows = 200
view_weights = np.asarray([0.7, .3])
cluster_weights = [np.array([.33, .33, .34]), np.array([.2, .8])]
cctypes = ['beta_uc', 'normal','normal_uc','poisson','categorical',
    'vonmises', 'bernoulli', 'lognormal']

separation = [.7] * 9
distargs = [None, None, None, None, {'k':5}, None, None, None, None]

T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation, return_dims=True)

runner = engine.Engine(T.T, cctypes, distargs, num_states=6, initialize=True)
runner.transition(N=2, multithread=True)

# state = State(T, cctypes, distargs)
# state.transition(N=10, do_plot=True)
