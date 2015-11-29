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

import numpy as np

from gpmcc.utils.validation import validate_cctypes
from gpmcc.experiments.particle_dim import ParticleDim

import multiprocessing

def _particle_learn(args):
    dim, X, seed = args
    np.random.seed(seed)
    np.random.shuffle(X)
    dim.particle_learn(X)
    return dim

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
        args = [(dim, X, seed) for (dim, seed) in zip(self.dims, seeds)]
        self.dims = self.map(_particle_learn, args)

    def get_dim(self, index):
        return self.dims[index]
