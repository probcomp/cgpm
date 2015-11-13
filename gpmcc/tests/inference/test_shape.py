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
    ct_kernel = args["ct_kernel"]

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
        pylab.suptitle( "Kernel %i" % ct_kernel)

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
            S = state.State(T, cctypes, ct_kernel=ct_kernel, distargs=distargs)
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
    # python shape_test_chains.py --num_chains 20 --num_rows 500 --ct_kernel 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', default=200, type=int)
    parser.add_argument('--num_rows', default=500, type=int)
    parser.add_argument('--num_chains', default=8, type=int)
    parser.add_argument('--ct_kernel', default=0, type=int)

    args = parser.parse_args()
    num_iters = args.num_iters
    num_chains = args.num_chains
    num_rows = args.num_rows
    ct_kernel = args.ct_kernel

    args = dict(num_rows=num_rows, num_iters=num_iters, num_chains=num_chains,
        ct_kernel=ct_kernel)
    run_test(args)
