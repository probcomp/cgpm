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

from gpmccc import state
from gpmcc.utils import general as gu

import numpy as np
import pylab

n_rows = 300
n_states = 32

distargs = [ {'K':n_rows} ]
distargs.extend( [None]*8 )
cctypes = ['multinomial']
cctypes.extend(['normal']*8)
column_names = ['id']
column_names.extend( ['one cluster']*4 )
column_names.extend( ['four cluster']*4 )

np.random.seed(10)

# id col
X = [np.array([i for i in range(n_rows)], dtype=int)]

# four cols of one cluster
for i in range(4):
    X.append( np.random.randn(n_rows) )

Z = []
for i in range(n_rows):
    Z.append(np.random.randrange(4))

# four cols of
for _ in range(4):
    x_clustered = []
    for i in range(n_rows):
        x_clustered.append( np.random.randn()+Z[i]*4 )

    X.append(np.array( x_clustered))

states = []
for s in range(n_states):
    states.append(state.State(X, cctypes, distargs,
        seed=np.random.randrange(200000)))

num_iters = 200
i = 0
for state in states:
    i += 1
    state.transition(N=num_iters)
    print "state %i of %i" % (i, n_states)

Zvs = []
for state in states:
    Zvs.append(state.Zv.tolist())

for Zv in Zvs:
    print Zv

fig = pylab.figure(num=None, figsize=(8,6),
            facecolor='w', edgecolor='k',frameon=False,
            tight_layout=True)

gu.generate_Z_matrix(Zvs, column_names)

# pylab.show()
