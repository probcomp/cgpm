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
import matplotlib.pyplot as plt

from cgpm.utils import test as tu
from cgpm.utils import sampling as su
from cgpm.crosscat.engine import Engine

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

        engine = Engine(
            T_o.T, cctypes=cctypes, distargs=distargs, num_states=n_chains)
        engine.transition(N=n_iters)

        for chain in xrange(n_chains):
            state = engine.get_state(chain)
            print "chain %i of %i" % (chain+1, n_chains)
            T_i.extend(state.simulate(-1, [0,1], N=n_per_chain))

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
