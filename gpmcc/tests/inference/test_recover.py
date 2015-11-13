# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
#
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

import numpy as np
import pylab

from gpmcc.utils import test as tu
from gpmcc.utils import sampling as su
from gpmcc import state

gen_function = {
    'sin' : tu.gen_sine_wave,
    'x' : tu.gen_x,
    'ring' : tu.gen_ring,
    'dots' : tu.gen_four_dots
}

cctypes = ['normal', 'normal']
distargs = [None, None]

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
            S = state.State(T_o, cctypes, distargs=distargs)
            S.transition(N=n_iters)

            T_i.extend(su.simple_predictive_sample(S, n_rows, [0,1],
                N=n_per_chain))

        T_i = np.array(T_i)
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
    args = dict(num_rows=1000, num_iters=200, num_chains=1)
    run_test(args)
