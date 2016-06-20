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
import pylab

from cgpm.crosscat.state import State
from cgpm.utils import test as tu

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
        S = state.State(Ts, cctypes=cctypes, distargs=distargs)
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
