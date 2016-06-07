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

from scipy.special import gammaln
from scipy.stats import t

import gpmcc.utils.general as gu

from gpmcc.exponentials.distribution import DistributionGpm


class Normal(DistributionGpm):
    """Normal distribution with normal prior on mean and gamma prior on
    precision. Collapsed.

    rho ~ Gamma(nu/2, s/2)
    mu ~ Normal(m, r*rho)
    x ~ Normal(mu, rho)

    http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
    Note that Teh uses Normal-InverseGamma to the **variance** has an inverse
    gamma distribution.
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

    def incorporate(self, rowid, query, evidence):
        DistributionGpm.incorporate(self, rowid, query, evidence)
        x = query[self.outputs[0]]
        self.N += 1.
        self.sum_x += x
        self.sum_x_sq += x*x
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_x -= x
        self.sum_x_sq -= x*x

    def logpdf(self, rowid, query, evidence):
        DistributionGpm.logpdf(self, rowid, query, evidence)
        x = query[self.outputs[0]]
        return Normal.calc_predictive_logp(
            x, self.N, self.sum_x, self.sum_x_sq, self.m, self.r,
            self.s, self.nu)

    def simulate(self, rowid, query, evidence, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        DistributionGpm.simulate(self, rowid, query, evidence)
        if rowid in self.data:
            return self.data[rowid]
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
        mn, rn, sn, nun = Normal.posterior_hypers(
            N, sum_x, sum_x_sq, m, r, s, nu)
        mm, rm, sm, num = Normal.posterior_hypers(
            N+1, sum_x+x, sum_x_sq+x*x, m, r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        ZM = Normal.calc_log_Z(rm, sm, num)
        return -.5 * np.log(2*np.pi) + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_x, sum_x_sq, m, r, s, nu):
        mn, rn, sn, nun = Normal.posterior_hypers(
            N, sum_x, sum_x_sq, m, r, s, nu)
        Z0 = Normal.calc_log_Z(r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        return -(N/2.) * np.log(2*np.pi) + ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + float(N)
        nun = nu + float(N)
        mn = (r*m + sum_x)/rn
        sn = s + sum_x_sq + r*m*m - rn*mn*mn
        if sn == 0: sn = s
        return mn, rn, sn, nun

    @staticmethod
    def calc_log_Z(r, s, nu):
        return (
            ((nu + 1.) / 2.) * np.log(2)
            + .5 * np.log(np.pi)
            - .5 * np.log(r)
            - (nu/2.) * np.log(s)
            + gammaln(nu/2.0))

    @staticmethod
    def posterior_logcdf(x, N, sum_x, sum_x_sq, m, r, s, nu):
        mn, rn, sn, nun = Normal.posterior_hypers(
            N, sum_x, sum_x_sq, m, r, s, nu)
        scalesq = sn/2.*(rn+1) / (nun/2.*rn)
        return t.logcdf(x, 2*nun/2., loc=mn, scale=np.sqrt(scalesq))

    @staticmethod
    def sample_parameters(m, r, s, nu, rng):
        rho = rng.gamma(nu/2., scale=2./s)
        mu = rng.normal(loc=m, scale=1./(rho*r)**.5)
        return mu, rho
