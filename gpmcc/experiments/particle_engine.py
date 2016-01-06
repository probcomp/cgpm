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

import numpy as np

from gpmcc.utils.validation import validate_cctypes
from gpmcc.experiments.particle_dim import ParticleDim

import multiprocessing

def _particle_learn(args):
    metadata, X, seed = args
    np.random.seed(seed)
    np.random.shuffle(X)
    dim = ParticleDim.from_metadata(metadata)
    dim.particle_learn(X, progress=True)
    return dim.to_metadata()

class ParticleEngine(object):
    """Particle Engine."""

    def __init__(self, dist, particles=1, distargs=None, multithread=True):
        validate_cctypes([dist])
        self.multithread = multithread
        self.map = map
        if self.multithread:
            self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.map = self.pool.map
        self.dist = dist
        self.distargs = distargs
        self.particles = particles
        self.dims = [ParticleDim(dist, distargs) for _ in xrange(particles)]

    def particle_learn(self, X, seeds=None):
        """Do particle learning in parallel."""
        if seeds is None:
            seeds = range(self.particles)
        assert len(seeds) == self.particles
        args = [(dim.to_metadata(), X, seed) for (dim, seed)
            in zip(self.dims, seeds)]
        metadata = self.map(_particle_learn, args)
        self.dims = [ParticleDim.from_metadata(m) for m in metadata]

    def get_dim(self, index):
        return self.dims[index]
