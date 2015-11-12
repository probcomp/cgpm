from baxcat.utils import cc_test_utils as tu
from baxcat.utils import cc_sample_utils as su
from baxcat import cc_state

import numpy
import pylab

gen_function = {
    'sin' : tu.gen_sine_wave,
    'x' : tu.gen_x,
    'ring' : tu.gen_ring,
    'dots' : tu.gen_four_dots
}

cctypes = ['normal']*2
distargs = [None]*2

shapes = ['x', 'sin', 'ring', 'dots']

def run_test(argsin):
    n_rows = args["num_rows"]
    n_iters = args["num_iters"]
    n_chains = args["num_chains"]

    n_per_chain = int(float(n_rows)/n_chains)

    plt = 0
    for shape in shapes:
        print "Shape: %s" % shape
        plt += 1
        T_o = gen_function[shape](n_rows)
        T_i = []
        for chain in range(n_chains):
            print "chain %i of %i" % (chain+1, n_chains)
            S = cc_state.cc_state(T_o, cctypes, ct_kernel=1, distargs=distargs)
            S.transition(N=n_iters)

            T_i.extend( su.simple_predictive_sample(S, n_rows, [0,1], N=n_per_chain) )

        T_i = numpy.array(T_i)

        ax = pylab.subplot(2,4,plt)
        pylab.scatter( T_o[0], T_o[1], color='blue', edgecolor='none' )
        pylab.ylabel("X")
        pylab.ylabel("Y")
        pylab.title("%s original" % shape)

        pylab.subplot(2,4,plt+4)
        pylab.scatter( T_i[:,0], T_i[:,1], color='red', edgecolor='none' )
        pylab.ylabel("X")
        pylab.ylabel("Y")
        pylab.xlim(ax.get_xlim())
        pylab.ylim(ax.get_ylim())
        pylab.title("%s simulated" % shape)

    print "Done."
    pylab.show()


if __name__ == "__main__":
    args = dict(num_rows=1000,num_iters=200,num_chains=1)
    run_test(args)