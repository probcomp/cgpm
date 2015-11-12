import baxcat.utils.cc_inference_utils as iu
from baxcat import cc_state
import numpy
import random
import pylab

rho_list = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
n_data_sets = 3
n_samples = 5


N = 50
mu = numpy.zeros(2)

i = 0

distargs = [None,None]

# for kernel in range(2):
for kernel in range(2):
    L = numpy.zeros((n_data_sets*n_samples, len(rho_list)))
    c = 0
    for rho in rho_list:
        r = 0
        for ds in range(n_data_sets):
            # seed control so that data is always the same
            numpy.random.seed(r+ds)
            random.seed(r+ds)

            sigma = numpy.array([[1,rho],[rho,1]])
            X = numpy.random.multivariate_normal(mu,sigma,N)

            for _ in range(n_samples):
                S = cc_state.cc_state([X[:,0], X[:,1]], ['normal']*2, Zv=[0,0],
                    ct_kernel=kernel, distargs=distargs)

                S.transition(N=100)

                MI = iu.mutual_information(S, 0, 1)
                linfoot = iu.mutual_information_to_linfoot(MI)

                # del S

                L[r,c] = linfoot

                print("rho: %1.2f, MI: %1.6f, Linfoot: %1.6f" %(rho, MI, linfoot))
                print("%i of %i" % (i+1, len(rho_list)*n_data_sets*n_samples*2))

                del S

                i += 1
                r += 1
        c += 1

    rho_labs = [str(rho) for rho in rho_list]


    ax = pylab.subplot(1,2,kernel+1)
    pylab.boxplot(L)
    pylab.ylim([0,1])
    pylab.ylabel('Linfoot')
    pylab.xlabel('rho')
    pylab.title("kernel %i" % kernel)
    ax.set_xticklabels(rho_labs)

pylab.show()