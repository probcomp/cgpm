import gpmcc.utils.inference_utils as iu
import gpmcc.utils.cc_test_utils as tu

from  baxcat import cc_state 

import numpy
import random
import math
import pylab

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

Ts, Zv, Zc = tu.gen_data_table(n_rows,
            numpy.array([.5,.5]), 
            [numpy.array([1./2]*2),
            numpy.array([1./5]*5)], 
            cctypes, 
            distargs, 
            [1.0]*n_cols)

 
for kernel in range(n_kernels):
    # for a set number of chains
    ARI_view = numpy.zeros((n_data_sets, n_transitions))
    ARI_cols = numpy.zeros((n_data_sets, n_transitions))

    for r in range(n_data_sets):
        S = cc_state.cc_state(Ts, cctypes, ct_kernel=kernel, distargs=distargs)
        for c in range(n_transitions):
            S.transition(N=1)

            # calucalte ARI
            ari_view = adjusted_rand_score(Zv, S.Zv.tolist())
            ari_cols = tu.column_average_ari(Zv, Zc, S)

            ARI_view[r,c] = ari_view
            ARI_cols[r,c] = ari_cols

        itr += 1
        print("itr %i of %i." % (itr, total_itr))

    ###
    pylab.subplot(2,n_kernels,kernel+1)
    pylab.plot(numpy.transpose(ARI_view))
    
    pylab.plot(numpy.mean(ARI_view,axis=0), color='black', linewidth=3)

    pylab.xlabel('transition')
    pylab.ylabel('ARI')
    pylab.title("ARI (columns) kernel %i" % kernel)
    pylab.ylim([0,1.1])

    #####
    pylab.subplot(2,n_kernels,kernel+n_kernels+1)
    pylab.plot(numpy.transpose(ARI_cols))
    
    pylab.plot(numpy.mean(ARI_cols,axis=0), color='black', linewidth=3)

    pylab.xlabel('transition')
    pylab.ylabel('ARI')
    pylab.title("ARI (rows) kernel %i" % kernel)
    pylab.ylim([0,1.1])

    print("ARI's for kernel %i" % kernel)
    print(ARI_view[:,n_transitions-1])

pylab.show()

