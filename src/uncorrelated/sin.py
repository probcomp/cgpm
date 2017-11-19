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

from scipy.integrate import dblquad
from scipy.integrate import quad

from scipy.stats import uniform

from cgpm.cgpm import CGpm
from cgpm.network.importance import ImportanceNetwork
from cgpm.uncorrelated.directed import DirectedXyGpm
from cgpm.uncorrelated.uniformx import UniformX
from cgpm.utils import general as gu


class SinY(CGpm):
    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        if rng is None:
            rng = gu.gen_rng(1)
        if outputs is None:
            outputs = [0]
        if inputs is None:
            inputs = [1]
        if noise is None:
            noise = .1
        self.rng = rng
        self.outputs = outputs
        self.inputs = inputs
        self.noise = noise
        self.uniform = uniform(scale=self.noise)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets == self.outputs
        assert inputs.keys() == self.inputs
        assert not constraints
        x = inputs[self.inputs[0]]
        noise = self.rng.uniform(high=self.noise)
        if np.cos(x) < 0:
            y = np.cos(x) + noise
        else:
            y = np.cos(x) - noise
        return {self.outputs[0]: y}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert targets.keys() == self.outputs
        assert inputs.keys() == self.inputs
        assert not constraints
        x = inputs[self.inputs[0]]
        y = targets[self.outputs[0]]
        if np.cos(x) < 0:
            return self.uniform.logpdf(y-np.cos(x))
        else:
            return self.uniform.logpdf(np.cos(x)-y)

class Sin(DirectedXyGpm):
    """Y = cos(X) + Noise."""

    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        DirectedXyGpm.__init__(
            self, outputs=outputs, inputs=inputs, noise=noise, rng=rng)
        self.x = UniformX(
            outputs=[self.outputs[0]], low=-1.5*np.pi, high=1.5*np.pi)
        self.y = SinY(
            outputs=[self.outputs[1]],
            inputs=[self.outputs[0]],
            noise=noise)
        self.network = ImportanceNetwork([self.x, self.y], rng=self.rng)

    # All further methods are here for historical reasons and are not invoked.
    # Should override simuate and logpdf from the importance network in
    # DirectedXyGpm to activate them.

    def logpdf_xy(self, x, y):
        if not self.D[0] <= x <= self.D[1]:
            return -float('inf')
        if np.cos(x) < 0 and not np.cos(x) <= y <= np.cos(x) + self.noise:
            return -float('inf')
        if np.cos(x) > 0 and not np.cos(x) - self.noise <= y <= np.cos(x):
            return -float('inf')
        return -np.log(self.D[1]-self.D[0]) - np.log(self.noise)

    def logpdf_x(self, x):
        if not self.D[0] <= x <= self.D[1]:
            return -float('inf')
        return -np.log(self.D[1]-self.D[0])

    def logpdf_y(self, y):
        if 0 <= y:
            length, overflow = self._valid_x(y)
            length += 2*overflow
        else:
            length, overflow = self._valid_x(-y)
            length = 2*length + overflow
        length *= 2
        return np.log(length) - np.log(self.noise) - np.log(self.D[1]-self.D[0])

    def logpdf_x_given_y(self, x, y):
        raise NotImplementedError

    def logpdf_y_given_x(self, y, x):
        raise NotImplementedError

    def mutual_information(self):
        def mi_integrand(x, y):
            return np.exp(self.logpdf_xy(x,y)) * \
                (self.logpdf_xy(x,y) - self.logpdf_x(x) - self.logpdf_y(y))
        return dblquad(
            lambda y, x: mi_integrand(x,y), self.D[0], self.D[1],
            self._lower_y, self._upper_y)

    def _valid_x(self, y):
        """Compute valid regions of x for y \in [0, 1], with overflow."""
        assert 0<=y<=1
        x_max = np.arccos(y)
        if y+self.noise < 1:
            x_min = np.arccos(y+self.noise)
        else:
            x_min = 0
        # compute overflow
        overflow = 0
        if y < self.noise:
            overflow = np.arccos(y-self.noise) - np.pi / 2
        return x_max - x_min, overflow

    def _lower_y(self, x):
        if np.cos(x) < 0:
            return np.cos(x)
        else:
            return np.cos(x) - self.noise

    def _upper_y(self, x):
        if np.cos(x) < 0:
            return np.cos(x) + self.noise
        else:
            return np.cos(x)

    def _sanity_test(self):
        # Marginal of x integrates to one.
        print quad(lambda x: np.exp(self.logpdf_x(x)), self.D[0], self.D[1])

        # Marginal of y integrates to one.
        print quad(lambda y: np.exp(self.logpdf_y(y)), -1 ,1)

        # Joint of x,y integrates to one; quadrature will fail for small noise.
        print dblquad(
            lambda y,x: np.exp(self.logpdf_xy(x,y)), self.D[0], self.D[1],
            lambda x: -1, lambda x: 1)
