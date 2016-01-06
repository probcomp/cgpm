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

import numpy as np
import pylab

from gpmcc.utils import test as tu
from gpmcc import state

from sklearn.metrics import adjusted_rand_score

n_rows = 100
n_cols = 16

n_transitions = 300
n_data_sets = 10 # the number of samples (chains)

n_kernels = 2

total_itr = n_kernels*n_data_sets
itr = 0

cctypes = ['normal']*n_cols
distargs = [None]*n_cols

Ts, Zv, Zc = tu.gen_data_table(n_rows,np.array([.5,.5]),
    [np.array([1./2]*2), np.array([1./5]*5)], cctypes, distargs, [1.0]*n_cols)

for kernel in range(n_kernels):
    # for a set number of chains
    ARI_view = np.zeros((n_data_sets, n_transitions))
    ARI_cols = np.zeros((n_data_sets, n_transitions))

    for r in range(n_data_sets):
        S = state.State(Ts, cctypes, distargs=distargs)
        for c in range(n_transitions):
            S.transition(N=1)

            # calucalte ARI
            ari_view = adjusted_rand_score(Zv, S.Zv.tolist())
            ari_cols = tu.column_average_ari(Zv, Zc, S)

            ARI_view[r,c] = ari_view
            ARI_cols[r,c] = ari_cols

        itr += 1
        print "itr %i of %i." % (itr, total_itr)

    ###
    pylab.subplot(2,n_kernels,kernel+1)
    pylab.plot(np.transpose(ARI_view))
    pylab.plot(np.mean(ARI_view, axis=0), color='black', linewidth=3)
    pylab.xlabel('transition')
    pylab.ylabel('ARI')
    pylab.title("ARI (columns) kernel %i" % kernel)
    pylab.ylim([0,1.1])
    #####
    pylab.subplot(2,n_kernels,kernel+n_kernels+1)
    pylab.plot(np.transpose(ARI_cols))
    pylab.plot(np.mean(ARI_cols, axis=0), color='black', linewidth=3)
    pylab.xlabel('transition')
    pylab.ylabel('ARI')
    pylab.title("ARI (rows) kernel %i" % kernel)
    pylab.ylim([0,1.1])
    print "ARI's for kernel %i" % kernel
    print ARI_view[:,n_transitions-1]

pylab.show()
