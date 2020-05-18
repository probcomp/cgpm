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

from __future__ import division
from past.utils import old_div
from math import lgamma
from math import log
from math import pi

import numpy as np

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu


LOG2 = log(2)
LOGPI = log(pi)
LOG2PI = LOG2 + LOGPI


class Normal(DistributionGpm):
    """Normal distribution with normal prior on mean and gamma prior on
    precision. Collapsed.

    rho ~ Gamma(nu/2, s/2)
    mu ~ Normal(m, r*rho)
    x ~ Normal(mu, rho)

    http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf Teh
    titles the document "Normal Inverse-Gamma Prior". In the description of the
    prior (Eq (3)) a Gamma distribution is used for precision \rho. Thus, if
    "\rho ~ Gamma" then "var = 1 / \rho ~ Inverse-Gamma".
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Sufficient statistics.
        self.N = 0
        self.sum_x = 0
        self.sum_x_sq = 0
        # Hyper parameters.
        if hypers is None: hypers = {}
        self.m = hypers.get('m', 0.)
        self.r = hypers.get('r', 1.)
        self.s = hypers.get('s', 1.)
        self.nu = hypers.get('nu', 1.)
        assert self.s > 0.
        assert self.r > 0.
        assert self.nu > 0.

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        self.N += 1.
        self.sum_x += x
        self.sum_x_sq += x*x
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_x -= x
        self.sum_x_sq -= x*x

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        return Normal.calc_predictive_logp(
            x, self.N, self.sum_x, self.sum_x_sq, self.m, self.r,
            self.s, self.nu)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        mn, rn, sn, nun = Normal.posterior_hypers(
            self.N, self.sum_x, self.sum_x_sq, self.m, self.r, self.s, self.nu)
        mu, rho = Normal.sample_parameters(mn, rn, sn, nun, self.rng)
        x = self.rng.normal(loc=mu, scale=rho**-.5)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Normal.calc_logpdf_marginal(
            self.N, self.sum_x, self.sum_x_sq, self.m, self.r, self.s, self.nu)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['r'] > 0.
        assert hypers['s'] > 0.
        assert hypers['nu'] > 0.
        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def get_hypers(self):
        return {'m': self.m, 'r': self.r, 's': self.s, 'nu': self.nu}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x, 'sum_x_sq': self.sum_x_sq}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        # Plus 1 for single observation case.
        N = len(X) + 1.
        ssqdev = np.var(X) * len(X) + 1.
        # Data dependent heuristics.
        grids['m'] = np.linspace(min(X), max(X) + 5, n_grid)
        grids['r'] = gu.log_linspace(1. / N, N, n_grid)
        grids['s'] = gu.log_linspace(ssqdev / 100., ssqdev, n_grid)
        grids['nu'] = gu.log_linspace(1., N, n_grid) # df >= 1
        return grids

    @staticmethod
    def name():
        return 'normal'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, sum_x_sq, m, r, s, nu):
        _mn, rn, sn, nun = Normal.posterior_hypers(
            N, sum_x, sum_x_sq, m, r, s, nu)
        _mm, rm, sm, num = Normal.posterior_hypers(
            N+1, sum_x+x, sum_x_sq+x*x, m, r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        ZM = Normal.calc_log_Z(rm, sm, num)
        return -.5 * LOG2PI + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, sum_x_sq, m, r, s, nu):
        _mn, rn, sn, nun = Normal.posterior_hypers(
            N, sum_x, sum_x_sq, m, r, s, nu)
        Z0 = Normal.calc_log_Z(r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        return -(N/2.) * LOG2PI + ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + float(N)
        nun = nu + float(N)
        mn = old_div((r*m + sum_x),rn)
        sn = s + sum_x_sq + r*m*m - rn*mn*mn
        if sn == 0:
            sn = s
        return mn, rn, sn, nun

    @staticmethod
    def calc_log_Z(r, s, nu):
        return (
            ((nu + 1.) / 2.) * LOG2
            + .5 * LOGPI
            - .5 * log(r)
            - (nu/2.) * log(s)
            + lgamma(nu/2.))

    @staticmethod
    def sample_parameters(m, r, s, nu, rng):
        rho = rng.gamma(nu/2., scale=2./s)
        mu = rng.normal(loc=m, scale=1./(rho*r)**.5)
        return mu, rho
