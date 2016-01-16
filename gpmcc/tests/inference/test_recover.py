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
import matplotlib.pyplot as plt

from gpmcc.utils import test as tu
from gpmcc.utils import sampling as su
from gpmcc.engine import Engine

shapes = ['x', 'sin', 'ring', 'dots']
gen_function = {
    'sin'   : tu.gen_sine_wave,
    'x'     : tu.gen_x,
    'ring'  : tu.gen_ring,
    'dots'  : tu.gen_four_dots
}

cctypes = ['normal', 'normal']
distargs = [None, None]


def run_test(args):
    n_rows = args["num_rows"]
    n_iters = args["num_iters"]
    n_chains = args["num_chains"]

    n_per_chain = int(float(n_rows)/n_chains)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,9))
    axes = axes.ravel()
    k = 0
    for shape in shapes:
        print "Shape: %s" % shape
        T_o = np.asarray(gen_function[shape](n_rows))
        T_i = []

        engine = Engine(T_o.T, cctypes, distargs, num_states=n_chains,
            initialize=True)
        engine.transition(N=n_iters)

        for chain in xrange(n_chains):
            state = engine.get_state(chain)
            print "chain %i of %i" % (chain+1, n_chains)
            T_i.extend(su.simulate(state, -1, [0,1], N=n_per_chain))

        T_i = np.array(T_i)

        ax = axes[k]
        ax.scatter( T_o[0], T_o[1], color='blue', edgecolor='none' )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("%s original" % shape)

        ax = axes[k+4]
        ax.scatter( T_i[:,0], T_i[:,1], color='red', edgecolor='none' )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.set_title("%s simulated" % shape)

        k += 1

    print "Done."
    return fig

if __name__ == "__main__":
    args = dict(num_rows=1000, num_iters=5000, num_chains=6)
    fig = run_test(args)
    fig.savefig('recover.png')
