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

from gpmcc.experiments.particle_dim import ParticleDim

import multiprocessing

def _particle_learn(args):
    X, cctype, distargs, seed = args
    np.random.seed(seed)
    np.random.shuffle(X)
    dim = ParticleDim(X, cctype, distargs)
    dim.particle_learn()
    return dim

class ParticleEngine(object):
    """Particle Engine."""

    def __init__(self, X, dist, distargs=None, multithread=True):
        self.multithread = multithread
        self.map = map
        if self.multithread:
            self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.map = self.pool.map
        self.X = X
        self.dist = dist
        self.distargs = distargs
        self.dims = None

    def particle_learn(self, particles=1, seeds=None):
        """Do particle learning in parallel."""
        if seeds is None:
            seeds = range(particles)
        assert len(seeds) == particles
        args = ((self.X, self.dist, self.distargs, seed) for (_, seed) in
            zip(xrange(particles), seeds))
        self.dims = self.map(_particle_learn, args)

    def get_dim(self, index):
        return self.dims[index]
