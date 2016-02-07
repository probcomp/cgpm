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

import math
from math import log, sqrt, pi

import numpy as np
from scipy.stats import norm, invgamma, uniform

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class NormalTrunc(DistributionGpm):
    """Truncated Normal distribution on range [l, h] with normal prior on mean
    and inverse gamma prior on variance. Uncollapsed.

    sigma2 ~ InverseGamma(a, b)
    mu ~ Normal(m, V*sigma2)
    x ~ NormalTrunc(mu, sigma2; l, h)

    http://fbe.unimelb.edu.au/__data/assets/pdf_file/0006/805866/856.pdf
    """

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, mu=None, sigma2=None, m=0.,
            V=1., a=1., b=1., distargs=None):
        assert V > 0.
        assert a > 0.
        assert b > 0.
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        self.sum_x_sq = sum_x_sq
        # Hyper parameters.
        self.m = float(m)
        self.V = float(V)
        self.a = float(a)
        self.b = float(b)
        # Distargs.
        self.l, self.h = distargs['l'], distargs['h']
        # Data for Gibbs.
        self.data = []
        # Parameters.
        self.mu, self.sigma2 = mu, sigma2
        if mu is None or sigma2 is None:
            sigma2 = invgamma.rvs(a, scale=b)
            mu = norm.rvs(loc=m, scale=sqrt(sigma2*V))

    def incorporate(self, x):
        assert self.l<x<self.h
        self.data.append(x)
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.data.remove(x)
        self.N -= 1.0
        if self.N == 0:
            self.sum_x = 0.0
            self.sum_x_sq = 0.0
        else:
            self.sum_x -= x
            self.sum_x_sq -= x*x

    def logpdf(self, x):
        return NormalTrunc.calc_predictive_logp(x, self.mu, self.sigma2,
            self.l, self.h)

    def logpdf_singleton(self, x):
        return NormalTrunc.calc_predictive_logp(x, self.mu, self.sigma2,
            self.l, self.h)

    def logpdf_marginal(self):
        logp_data = NormalTrunc.calc_log_likelihood(self.N, self.sum_x,
            self.sum_x_sq, self.mu, self.sigma2, self.l, self.h)
        logp_prior = NormalTrunc.calc_log_prior(self.mu, self.sigma2, self.m,
            self.V, self.a, self.b)
        return logp_prior + logp_data

    def simulate(self):
        u = uniform.rvs()
        sigma = sqrt(self.sigma2)
        term1 = norm.cdf((self.l-self.mu)/sigma)
        term2 = norm.cdf((self.h-self.mu)/sigma)
        return self.mu + sigma*norm.ppf(term1+u*(term2-term1))

    def transition_params(self):
        n_samples = 25
        for _ in xrange(n_samples):
            ybar, ydev = NormalTrunc.compute_auxiliary_data(
                self.data, self.mu, self.sigma2, self.l, self.h)
            self.mu = norm.rvs(loc=ybar, scale=sqrt(self.sigma2/self.N))
            self.sigma2 = invgamma.rvs(self.N/2., scale=-.5*ydev)

    def set_hypers(self, hypers):
        assert hypers['V'] > 0.
        assert hypers['a'] > 0.
        assert hypers['b'] > 0.
        self.m = hypers['m']
        self.V = hypers['V']
        self.a = hypers['a']
        self.b = hypers['b']

    def get_hypers(self):
        return {'m': self.m, 'V': self.V, 'a': self.a, 'b': self.b}

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x, 'sum_x_sq': self.sum_x_sq}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = len(X) + 1.
        ssqdev = np.var(X) * len(X) + 1.
        # Data dependent heuristics.
        grids['m'] = np.linspace(min(X)-5, max(X)+5, n_grid)
        grids['V'] = gu.log_linspace(1. / N, N, n_grid)
        grids['a'] = gu.log_linspace(1., N, n_grid) # df >= 1
        grids['b'] = gu.log_linspace(ssqdev / 100., ssqdev, n_grid)
        return grids

    @staticmethod
    def name():
        return 'normal_trunc'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, mu, sigma2, l, h):
        if not l < x < h:
            return -float('inf')
        log_data = norm.logpdf(x, loc=mu, scale=sqrt(sigma2))
        log_normalizer = norm.logcdf(h, loc=mu, scale=sqrt(sigma2))\
            - norm.logcdf(l, loc=mu, scale=sqrt(sigma2))
        return log_data - log_normalizer

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, mu, sigma2, l, h):
        log_data = -(N/2.)*log(2*pi) - (N/2.)*log(sigma2) - \
            1./(2*sigma2)*(sum_x_sq - 2*mu*sum_x + N*mu*mu)
        log_normalizer = norm.logcdf(h, loc=mu, scale=sqrt(sigma2))\
            - norm.logcdf(l, loc=mu, scale=sqrt(sigma2))
        return log_data - N * log_normalizer

    @staticmethod
    def calc_log_prior(mu, sigma2, m, V, a, b):
        log_sigma2 = invgamma.logpdf(sigma2, a, scale=b)
        log_mu = norm.logpdf(mu, loc=m, scale=np.sqrt(sigma2*V))
        return log_sigma2 + log_mu

    @staticmethod
    def compute_auxiliary_data(data, mu, sigma2, l, h):
        sum_y, sum_y_sq = 0., 0.
        sigma = sqrt(sigma2)
        low = norm.cdf((h-mu)/sigma) - norm.cdf((l-mu)/sigma)
        for x in data:
            top = norm.cdf((x-mu)/sigma) - norm.cdf((l-mu)/sigma)
            y =  mu + sigma* norm.ppf(top/low)
            sum_y += y
            sum_y_sq += y*y
        ybar = sum_y/len(data)
        ydev = sum_y_sq -2*mu*sum_y + len(data)*mu*mu
        return ybar, ydev
