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
from particle_engine import ParticleEngine

def logmeanexp(values):
    return logsumexp(values) - np.log(len(values))

# Fix seed for random dataset.
np.random.seed(10)

# Set up the data generation.
n_rows = 100
view_weights = np.ones(1)
cluster_weights = [np.asarray([.5, .3, .2])]
dists = ['normal']
separation = [.8]
distargs = [None]
T, Zv, Zc,= tu.gen_data_table(n_rows, view_weights, cluster_weights, dists,
    distargs, separation)

# Initialize the engine.
num_particles = 6
engine = ParticleEngine('normal', particles=num_particles)
engine.particle_learn(T[0])

# Obtain the marginal logps from each chain.
weights = [dim.weight for dim in engine.dims]

# Compute the estimator of logZ.
logZ = logmeanexp(weights)
