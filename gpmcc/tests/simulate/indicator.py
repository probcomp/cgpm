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

"""
This graphical test trains a gpmcc state on a bivariate population [X, Z] where
X is normally distributed data and Z is a (transformed) latent cluster
assignment for each row. We then simulate from the joint posterior distribution
of gpmcc, and compare the simulations to the ingested data at different
subpopulations of X (which are identified by the Z).
"""

import math
import numpy as np
import pylab
import matplotlib.pyplot as plt

import gpmcc.utils.sampling as su
import gpmcc.utils.sampling as tu
import gpmcc.utils.general as gu
from gpmcc.engine import Engine

def test_simulate_indicator():
    # Entropy.
    np.random.seed(0)
    # Generate synthetic dataset.
    n_rows = 250
    mus = [-1, 5, 9]
    sigmas = [2, 2, 1.5]
    data = np.zeros((n_rows, 2))
    indicators = [0, 1, 2, 3, 4, 5]
    data[:, 1] = np.random.choice(indicators, size=n_rows,
        p=[.15, .15, .25, .25, .1, .1])
    for i in xrange(n_rows):
        idx = int(data[i,1] / 2)
        data[i, 0] = np.random.normal(loc=mus[idx], scale=sigmas[idx])

    # Create an engine.
    state = Engine(data, ['normal', 'categorical'], [None, {'k':6}],
        num_states=1, initialize=True)
    state.transition(N=150)
    model = state.get_state(0)

    # Simulate from the joint distribution of (x,i).
    samples = su.simulate(model, -1, [0, 1], N=100)

    # Scatter the data points by color.
    fig, ax = plt.subplots()
    for t in indicators:
        # Plot original data.
        data_subpop = data[data[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data.
        samples_subpop = samples[samples[:,1] == t]
        ax.scatter(samples_subpop[:,1] + .25, samples_subpop[:,0],
            color=gu.colors[t])
    ax.set_xlabel('Indicator')
    ax.set_ylabel('x')
    ax.grid()
    plt.show()
