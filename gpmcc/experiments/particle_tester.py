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
    ParticleDim('exponential_uc'),
    ParticleDim('lognormal'),
    ParticleDim('beta_uc')
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
        dims[i].gibbs_transition()
        ax.clear()
        dims[i].plot_dist(ax=ax, Y=np.linspace(0.01,0.99,200))
        ax.grid()
        plt.draw()
        plt.pause(1.5)

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
