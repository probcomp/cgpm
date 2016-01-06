# -*- coding: utf-8 -*-

# The MIT License (MIT)

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

import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import logsumexp

from gpmcc.utils import test as tu
from gpmcc.utils import general as gu
from particle_dim import ParticleDim

# Fix the dataset.
np.random.seed(100)

# Competing densities.
dims = [
    ParticleDim('normal'),
    # ParticleDim('exponential_uc'),
    # ParticleDim('beta_uc'),
    ]

def logmeanexp(values):
    return logsumexp(values) - np.log(len(values))

def _gibbs_transition(args):
    dim = args
    dim.gibbs_transition()
    return dim

# Multiprocessor.
pool = multiprocessing.Pool(multiprocessing.cpu_count())

def observe_data(t):
    global dims
    global pool
    for d in dims:
        d.particle_learn([t])
    weights = [d.weight for d in dims]
    print 'Observation %f: %f' % (dims[0].Nobs, t)
    print weights
    while True:
        dims = pool.map(_gibbs_transition, ((d) for d in dims))
        i = gu.log_pflip(weights)
        # dims[i].gibbs_transition()
        ax.clear()
        dims[i].plot_dist(ax=ax, Y=np.linspace(0.01,0.99,200))
        ax.grid()
        plt.draw()
        plt.pause(.3)

def on_click(event):
    if event.button == 1:
        if event.inaxes is not None:
            observe_data(event.xdata)

# Activate the plotter.
plt.ion(); plt.show()
_, ax = plt.subplots()
plt.connect('button_press_event', on_click)

# Data generation.
n_rows = 100
view_weights = np.ones(1)
cluster_weights = [ np.array([.5, .5]) ]
cctypes = ['beta_uc']
separation = [.8]
distargs = [None]
T, Zv, Zc = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation)

# Test against synthetic data.
# for t in T[0]:
#     observe_data(t)
