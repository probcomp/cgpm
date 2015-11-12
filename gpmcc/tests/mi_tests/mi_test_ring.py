from baxcat.utils import cc_inference_utils as iu
import numpy
import random
from baxcat import cc_state
import math
import pylab

W_list = [.9, .75, .5, .25, .1]
n_data_sets = 3
n_samples = 5

N = 250
mu = numpy.zeros(2)

i = 0

def _gen_ring(n,w):

    X = numpy.zeros((n,2))
    for i in range(n):
        angle = random.uniform(0,2*math.pi)
        distance = random.uniform(1-w,1)
        X[i,0] = math.cos(angle)*distance
        X[i,1] = math.sin(angle)*distance

    return X+100


for kernel in range(2):
    MI = numpy.zeros((n_data_sets*n_samples, len(W_list)))
    c = 0
    for w in W_list:
        r = 0
        for ds in range(n_data_sets):
            # seed control so that data is always the same
            numpy.random.seed(r+ds)
            random.seed(r+ds)

            X = _gen_ring(N,w)


            for _ in range(n_samples):

                S = cc_state.cc_state([X[:,0], X[:,1]], ['normal']*2, ct_kernel=kernel, distargs=[None]*2)
                S.transition(N=200)

                mi = iu.mutual_information(S, 0, 1)
                # linfoot = iu.mutual_information_to_linfoot(MI)

                MI[r,c] = mi

                print("w: %1.2f, MI: %1.6f" % (w, mi))
                print("%i of %i" % (i+1, len(W_list)*n_data_sets*n_samples*2))

                del S

                i += 1
                r += 1
        c += 1

    w_labs = [str(w) for w in W_list]


    ax = pylab.subplot(1,2,kernel+1)
    pylab.boxplot(MI)
    pylab.ylim([0,1])
    pylab.ylabel('MI')
    pylab.xlabel('ring width')
    pylab.title("kernel %i" % kernel)
    ax.set_xticklabels(w_labs)

pylab.show()