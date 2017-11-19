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

from cgpm.cgpm import CGpm
from cgpm.utils.general import gen_rng


class DirectedXyGpm(CGpm):
    """Interface directed two-dimensional GPMs over the R2 plane."""

    def __init__(self, outputs=None, inputs=None, noise=None, acc=1, rng=None):
        """Initialize the Gpm with given noise parameter.

        Parameters
        ----------
        noise : float
            Value in (0,1) indicating the noise level of the distribution.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        if type(self) is DirectedXyGpm:
            raise Exception('Cannot directly instantiate DirectedXyGpm.')
        if rng is None:
            rng = gen_rng(0)
        if outputs is None:
            outputs = [0, 1]
        if noise is None:
            noise = .1
        self.rng = rng
        self.outputs = outputs
        self.noise = noise
        self.acc = acc
        # Override the network in subclass.
        self.network = None

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        if self.network is None:
            raise ValueError('self.network not defined by %s' % (type(self),))
        return self.network.logpdf(rowid, targets, inputs)

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        if self.network is None:
            raise ValueError('self.network not defined by %s' % (type(self),))
        return self.network.simulate(rowid, targets, inputs, N)
