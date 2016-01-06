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

from gpmcc.utils import test as tu
from gpmcc.utils import sampling as su
from gpmcc import state

import numpy
import pylab
import argparse

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

    fig = pylab.figure(num=None, facecolor='w', edgecolor='k',frameon=False,
        tight_layout=True)

    plt = 0
    data = {'x':[], 'sin':[], 'ring':[], 'dots':[]}
    xlims = dict()
    ylims = dict()
    for shape in shapes:
        plt += 1
        data[shape] = gen_function[shape](n_rows)

        ax = pylab.subplot(n_chains+1,4,plt)
        pylab.scatter( data[shape][0], data[shape][1], s=10, color='blue',
            edgecolor='none', alpha=.2 )
        # pylab.ylabel("X")
        # pylab.ylabel("Y")
        # pylab.title("%s original" % shape)

        ax.set_xticks([])
        ax.set_yticks([])

        xlims[shape] = ax.get_xlim()
        ylims[shape] = ax.get_ylim()

    States = []
    for chain in range(n_chains):
        print("chain %i of %i." % (chain+1, n_chains))
        plt = 0
        for shape in shapes:
            print("\tWorking on %s." % shape)
            plt += 1
            T = data[shape]
            S = state.State(T, cctypes, distargs=distargs)
            S.transition(N=n_iters)
            T_chain = numpy.array(su.simple_predictive_sample(S, n_rows, [0,1],
                N=n_rows))
            ax = pylab.subplot(n_chains+1,4,chain*4+4+plt)
            ax.set_xticks([])
            ax.set_yticks([])
            pylab.scatter( T_chain[:,0], T_chain[:,1], s=10, color='red',
                edgecolor='none', alpha=.2 )
            pylab.xlim(xlims[shape])
            pylab.ylim(ylims[shape])
            # pylab.title("%s simulated (%i)" % (shape, chain))

    print "Done."
    pylab.show()

if __name__ == "__main__":
    # python shape_test_chains.py --num_chains 20 --num_rows 500
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--num_rows', default=500, type=int)
    parser.add_argument('--num_chains', default=8, type=int)

    args = parser.parse_args()
    num_iters = args.num_iters
    num_chains = args.num_chains
    num_rows = args.num_rows

    args = dict(num_rows=num_rows, num_iters=num_iters, num_chains=num_chains)
    run_test(args)
