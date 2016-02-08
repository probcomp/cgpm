# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
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

from math import log

import numpy as np
from scipy.stats import norm, gamma

from gpmcc.utils import general as gu
from gpmcc.utils import sampling as su
from gpmcc.dists.normal import Normal

LOG2 = log(2.0)
LOGPI = log(np.pi)
LOG2PI = log(2*np.pi)

class NormalTrunc(Normal):
    """Normal distribution with normal prior on mean and gamma prior on
    precision. Uncollapsed.
    rho ~ Gamma(nu/2, s/2)
    mu ~ Normal(m, rho)
    X ~ Normal(mu, r*rho)
    http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
    Note that Teh uses Normal-InverseGamma to mean Normal-Gamma for prior.
    """

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, mu=None, rho=None, m=0,
            r=1, s=1, nu=1, distargs=None):
        # Invoke parent.
        super(NormalTrunc, self).__init__(N=N, sum_x=sum_x, sum_x_sq=sum_x_sq,
            m=m, r=r, s=s, nu=nu, distargs=distargs)
        # Distargs
        self.l = distargs['l']
        self.h = distargs['h']
        # Uncollapsed mean and precision parameters.
        self.mu, self.rho = mu, rho
        if mu is None or rho is None:
            self.mu, self.rho = Normal.sample_parameters(self.m, self.r, self.s,
                self.nu)
            # self.transition_params()

    def predictive_logp(self, x):
        return NormalTrunc.calc_predictive_logp(x, self.mu, self.rho) - \
            - NormalTrunc.calc_log_normalizer(self.mu, self.rho, self.l, self.h)

    def marginal_logp(self):
        data_logp = NormalTrunc.calc_log_likelihood(self.N, self.sum_x,
            self.sum_x_sq, self.rho, self.mu)
        prior_logp = NormalTrunc.calc_log_prior(self.mu, self.rho, self.m,
            self.r, self.s, self.nu)
        normalizer_logp = NormalTrunc.calc_log_normalizer(self.mu, self.rho,
            self.l, self.h)
        return data_logp + prior_logp - self.N * normalizer_logp

    def singleton_logp(self, x):
        return NormalTrunc.calc_predictive_logp(x, self.mu, self.rho)

    def simulate(self):
        return np.random.normal(self.mu, 1./self.rho**.5)

    def transition_params(self):
        n_samples = 25
        # Transition mu.
        log_pdf_fun_mu = lambda mu :\
            NormalTrunc.calc_log_likelihood(self.N, self.sum_x, self.sum_x_sq,
                self.rho, mu) \
            + NormalTrunc.calc_log_prior(mu, self.rho, self.m, self.r, self.s,
                self.nu) \
            - NormalTrunc.calc_log_normalizer(mu, self.rho, self.l, self.h)

        self.mu = su.mh_sample(self.mu, log_pdf_fun_mu,
            5, [-float('inf'), float('inf')], burn=n_samples)

        # Transition balance.
        log_pdf_fun_rho = lambda rho :\
            NormalTrunc.calc_log_likelihood(self.N, self.sum_x, self.sum_x_sq,
                rho, self.mu) \
            - NormalTrunc.calc_log_prior(self.mu, rho, self.m, self.r, self.s,
                self.nu) \
            - NormalTrunc.calc_log_normalizer(self.mu, rho, self.l, self.h)

        self.rho = su.mh_sample(self.rho, log_pdf_fun_rho,
            5, [0, float('inf')], burn=n_samples)

        # mn, rn, sn, nun = NormalTrunc.posterior_hypers(self.N, self.sum_x,
        #     self.sum_x_sq, self. m, self.r, self.s, self.nu)
        # self.mu, self.rho = NormalTrunc.sample_parameters(mn, rn, sn, nun)

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        # Plus 1 for single observation case.
        N = len(X) + 1.
        ssqdev = np.var(X) + 1.
        # Data dependent heuristics.
        grids['m'] = np.linspace(min(X)-5, max(X) + 5, n_grid)
        grids['r'] = gu.log_linspace(1. / N, N, n_grid)
        grids['s'] = gu.log_linspace(ssqdev / 100., ssqdev, n_grid)
        grids['nu'] = gu.log_linspace(1., N, n_grid) # df >= 1
        return grids

    @staticmethod
    def name():
        return 'normal_trunc'

    @staticmethod
    def is_collapsed():
        return False

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_log_normalizer(mu, rho, l, h):
        return (norm.logcdf(h, loc=mu, scale=rho**-.5) -
            norm.logcdf(l, loc=mu, scale=rho**-.5))

    @staticmethod
    def calc_predictive_logp(x, mu, rho):
        return norm.logpdf(x, loc=mu, scale=1./rho**.5)

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, rho, mu):
        return -(N / 2.) * LOG2PI + (N / 2.) * log(rho) - \
            .5 * (rho * (N * mu * mu - 2 * mu * sum_x + sum_x_sq))

    @staticmethod
    def calc_log_prior(mu, rho, m, r, s, nu):
        """Distribution of parameters (mu rho) ~ NG(m, r, s, nu)"""
        log_rho = gamma.logpdf(rho, nu/2., scale=2./s)
        log_mu = norm.logpdf(mu, loc=m, scale=1./(r*rho)**.5)
        return log_mu + log_rho
