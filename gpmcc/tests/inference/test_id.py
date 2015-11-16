# -*- coding: utf-8 -*-

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
