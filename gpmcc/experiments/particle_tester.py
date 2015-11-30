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

from gpmcc.utils import test as tu
from gpmcc.utils import general as gu
from particle_engine import ParticleEngine
from particle_dim import ParticleDim


def logmeanexp(values):
    from scipy.misc import logsumexp
    return logsumexp(values) - np.log(len(values))

# Fix the dataset.
np.random.seed(100)
particles = 20

# set up the data generation
n_rows = 10
view_weights = np.ones(1)
cluster_weights = [ np.array([.5, .5]) ]
cctypes = ['beta_uc']
separation = [.8]
distargs = [None]

# T = np.random.normal(loc=1000, scale=10, size=100)

def _gibbs_transition(args):
    dim = args
    dim.gibbs_transition()
    return dim

pool = multiprocessing.Pool(multiprocessing.cpu_count())

T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation, return_dims=True)
# dim.particle_learn(T[0])

dims = [ParticleDim('normal'), ParticleDim('exponential_uc'),
    ParticleDim('lognormal'), ParticleDim('beta_uc')]
dim = dims[0]

def observe_datum(t):
    for d in dims:
        d.particle_learn([t])
    return [d.weight for d in dims]

def observe_data(t):
    global dims
    weights = observe_datum(t)
    print 'Observation %f: %f' % (dim.Nobs, t)
    print weights
    ax.clear()
    while True:
        args = ((d) for d in dims)
        dims = pool.map(_gibbs_transition, args)
        i = gu.log_pflip(weights)
        dims[i].gibbs_transition()
        ax.clear()
        dims[i].plot_dist(ax=ax, Y=np.linspace(0.01,0.99,200))
        ax.grid()
        plt.draw()
        plt.pause(0.5)

plt.ion(); plt.show()
_, ax = plt.subplots()
def on_click(event):
    global dims
    # get the x and y coords, flip y from top to bottom
    if event.button == 1:
        if event.inaxes is not None:
            observe_data(event.xdata)

plt.connect('button_press_event', on_click)
