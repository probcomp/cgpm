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
import scipy

from gpmcc.dists.normal import Normal

LOG2 = log(2.0)
LOGPI = log(np.pi)
LOG2PI = log(2*np.pi)

class NormalUC(Normal):
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
        super(NormalUC, self).__init__(N=N, sum_x=sum_x, sum_x_sq=sum_x_sq,
            m=m, r=r, s=s, nu=nu, distargs=distargs)
        # Uncollapsed mean and precision parameters.
        self.mu, self.rho = mu, rho
        if mu is None or rho is None:
            self.transition_params()

    def predictive_logp(self, x):
        return NormalUC.calc_predictive_logp(x, self.mu, self.rho)

    def marginal_logp(self):
        data_logp = NormalUC.calc_log_likelihood(self.N, self.sum_x,
            self.sum_x_sq, self.rho, self.mu)
        prior_logp = NormalUC.calc_log_prior(self.mu, self.rho, self.m,
            self.r, self.s, self.nu)
        return data_logp + prior_logp

    def singleton_logp(self, x):
        return NormalUC.calc_predictive_logp(x, self.mu, self.rho)

    def simulate(self):
        return np.random.normal(self.mu, 1./self.rho**.5)

    def transition_params(self):
        rn, nun, mn, sn = NormalUC.posterior_hypers(self.N, self.sum_x,
            self.sum_x_sq, self. m, self.r, self.s, self.nu)
        self.rho = np.random.gamma(nun/2., scale=2./sn)
        self.mu = np.random.normal(loc=mn, scale=1./(self.rho*rn)**.5)

    @staticmethod
    def name():
        return 'normal_uc'

    @staticmethod
    def is_collapsed():
        return False

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, mu, rho):
        return scipy.stats.norm.logpdf(x, loc=mu, scale=1./rho**.5)

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, rho, mu):
        return -(N / 2.) * LOG2PI + (N / 2.) * log(rho) - \
            .5 * (rho * (N * mu * mu - 2 * mu * sum_x + sum_x_sq))

    @staticmethod
    def calc_log_prior(mu, rho, m, r, s, nu):
        """Distribution of parameters (mu rho) ~ NG(m, r, s, nu)"""
        log_rho = scipy.stats.gamma.logpdf(rho, nu/2., scale=2./s)
        log_mu = scipy.stats.norm.logpdf(mu, loc=m, scale=1./(r*rho)**.5)
        return log_mu + log_rho
