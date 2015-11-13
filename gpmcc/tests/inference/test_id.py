from baxcat import cc_state 
from shared_utils import general_utils as gu


import numpy
import random
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


random.seed(10)
numpy.random.seed(10)

# id col
X = [ numpy.array([i for i in range(n_rows)], dtype=int) ]

# four cols of one cluster
for i in range(4):
    X.append( numpy.random.randn(n_rows) )

Z = []
for i in range(n_rows):
    Z.append(random.randrange(4))

# four cols of 
for _ in range(4):
    x_clustered = []
    for i in range(n_rows):
        x_clustered.append( numpy.random.randn()+Z[i]*4 )

    X.append( numpy.array( x_clustered) )


# S = cc_state.cc_state(X, cctypes, distargs, ct_kernel=0, seed=random.randrange(200000))
# S.transition(N=200, do_plot=True)

states = []
for s in range(n_states):
    states.append( cc_state.cc_state(X, cctypes, distargs, ct_kernel=1, seed=random.randrange(200000)) )


num_iters = 200

i = 0
for state in states:
    i += 1
    state.transition(N=num_iters)
    print("state %i of %i" % (i, n_states) )


Zvs = []
for state in states: 
    Zvs.append(state.Zv.tolist())


for Zv in Zvs:
    print(Zv)

fig = pylab.figure(num=None, figsize=(8,6),
            facecolor='w', edgecolor='k',frameon=False, 
            tight_layout=True)



gu.generate_Z_matrix(Zvs, column_names)


# pylab.show()