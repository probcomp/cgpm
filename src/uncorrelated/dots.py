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

from scipy.stats import norm

from cgpm.uncorrelated.undirected import UnDirectedXyGpm
from cgpm.utils import general as gu


class Dots(UnDirectedXyGpm):
    """(X,Y) ~ Four Dots."""

    mx = [ -1, 1, -1, 1]
    my = [ -1, -1, 1, 1]

    def simulate_joint(self):
        n = self.rng.randint(4)
        x = self.rng.normal(loc=self.mx[n], scale=self.noise)
        y = self.rng.normal(loc=self.my[n], scale=self.noise)
        return [x, y]

    def simulate_conditional(self, z):
        return self.simulate_joint()[0]

    def logpdf_joint(self, x, y):
        return gu.logsumexp([np.log(.25)
                + norm.logpdf(x, loc=mx, scale=self.noise)
                + norm.logpdf(y, loc=my, scale=self.noise)
            for (mx,my) in zip(self.mx, self.my)])

    def logpdf_marginal(self, z):
        return gu.logsumexp(
            [np.log(.5) + norm.logpdf(z, loc=mx, scale=self.noise)
            for mx in set(self.mx)])

    def logpdf_conditional(self, w, z):
        return self.logpdf_marginal(z)
