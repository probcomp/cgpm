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
from sklearn.linear_model import Ridge

import gpmcc.utils.config as cu
import gpmcc.utils.data as du
from gpmcc.dists.distribution import DistributionGpm

class LinearRegression(DistributionGpm):
    """LinearRegression conditional distribution p(x|y) with L1
    regulariation. Note that x is the regressed variable and y are the
    covariates."""

    def __init__(self, distargs=None):
        p = 0
        self.discrete_covariates = {}
        for i, (cct, cca) in \
                enumerate(zip(distargs['cctypes'], distargs['ccargs'])):
            if cu.cctype_class(cct).is_numeric():
                p += 1
            elif 'k' in cca: # XXX HACK
                self.discrete_covariates[i] = cca['k']
                p += cca['k'] - 1
        # Observations.
        self.x = []
        self.Y = []
        # Number of covariates with dummy.
        self.p = p
        # Hyperparam.
        self.alpha = 1.0
        # Param.
        self.sigma = 0
        self.regressor = Ridge(alpha=self.alpha)

    def incorporate(self, x, y=None):
        y = du.dummy_code(y, self.discrete_covariates)
        assert len(y) == self.p
        self.Y.append(y)
        self.x.append(x)
        # XXX Do we need to retrain?
        self.regressor.fit(self.Y, self.x)
        self.sigma = np.sqrt(np.sum(
            (self.regressor.predict(self.Y) - self.x)**2 / len(self.x)))

    def unincorporate(self, x, y=None):
        y = du.dummy_code(y, self.discrete_covariates)
        assert len(y) == self.p
        # Search for y.
        for i in xrange(len(self.Y)):
            if self.Y[i] == y and self.x[i] == x:
                del self.Y[i]
                del self.x[i]
                break
        else:
            raise ValueError('Observation %s not incorporated.' % str((x, y)))

    def logpdf(self, x, y=None):
        # XXX We need to be sampling \beta_i from N(0,alpha) to compute a
        # singleton predictive, but the \beta_i are deep in linear_model.Ridge.
        if len(self.Y) == 0:
            return 0
        y = du.dummy_code(y, self.discrete_covariates)
        xhat = self.regressor.predict(y)
        error = x - xhat
        return norm.logpdf(error, scale=self.sigma)[0]

    def logpdf_marginal(self):
        if len(self.Y) == 0:
            return 0
        error = self.regressor.predict(self.Y) - self.x
        return np.sum(norm.logpdf(error, scale=self.sigma))

    def simulate(self, y=None):
        y = du.dummy_code(y, self.discrete_covariates)
        assert len(y) == self.p
        return self.regressor.predict(y)[0]

    def transition_params(self):
        self.regressor.fit(self.Y, self.x)
        self.sigma = np.sqrt(np.sum(
            (self.regressor.predict(self.Y) - self.x)**2 / len(self.x)))

    def set_hypers(self, hypers):
        return

    def get_hypers(self):
        return {}

    def get_params(self):
        """Return a dictionary of parameters."""
        return {}

    def get_suffstats(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=20):
        return {}

    @staticmethod
    def name():
        return 'linear_regression'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return True

    @staticmethod
    def is_numeric():
        return True

    # XXX Disabled.
    @staticmethod
    def calc_log_prior(beta, alpha):
        return np.sum(norm.logpdf(beta), scale=np.sqrt(alpha)**-1)
