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
from cgpm.utils import mvnormal as multivariate_normal


class Linear(UnDirectedXyGpm):

    def simulate_joint(self):
        return self.rng.multivariate_normal(
            [0,0], [[1,1-self.noise],[1-self.noise,1]])

    def simulate_conditional(self, z):
        mean = self.conditional_mean(z)
        var = self.conditional_variance(z)
        return self.rng.normal(loc=mean, scale=np.sqrt(var))

    def logpdf_joint(self, x, y):
        return multivariate_normal.logpdf(
            np.array([x,y]), np.array([0,0]),
            np.array([[1,1-self.noise],[1-self.noise,1]]))

    def logpdf_marginal(self, z):
        return norm.logpdf(z, scale=1)

    def logpdf_conditional(self, w, z):
        mean = self.conditional_mean(z)
        var = self.conditional_variance(z)
        return norm.logpdf(w, loc=mean, scale=np.sqrt(var))

    def conditional_mean(self, z):
        return (1-self.noise)*z

    def conditional_variance(self, z):
        return (1-(1-self.noise)**2)

    def mutual_information(self):
        cov = 1-self.noise
        return -.5 * np.log(1-cov**2)
