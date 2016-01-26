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


from gpmcc.utils import config as cu
from gpmcc.utils import test as tu
from gpmcc import state
import numpy

# This script generates acolumn of every data type and plots inference
# in real time

numpy.random.seed(10)

# set up the data generation
n_rows = 200
view_weights = numpy.ones(1)
cluster_weights = [ numpy.array([.33, .33, .34]) ]
cctypes = ['normal', 'poisson', 'bernoulli', 'lognormal', 'exponential',
    'normal_uc', 'beta_uc', 'exponential_uc', 'geometric', 'vonmises']
separation = [.95] * len(cctypes)
cctypes, distargs = cu.parse_distargs(cctypes)

T, Zv, Zc = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation)

S = state.State(T.T, cctypes, distargs, seed=0)
S.transition(N=30)
S.plot()
