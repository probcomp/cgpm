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

from math import atan2, cos, log, pi, sin

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0 as bessel_0, i1 as bessel_1

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

TWOPI = 2*pi
LOG2PI = log(2*pi)

class Vonmises(DistributionGpm):
    """Von Mises distribution on [0, 2pi] with Vonmises prior on the mean
    mu. The concentration k is fixed at construct time (defaults to 1.5).

    mu ~ Vonmises(mean=b, concentration=a)
    x ~ Vonmises(mean=mu, concentration=k)
    """

    cctype = 'vonmises'

    def __init__(self, N=0, sum_sin_x=0, sum_cos_x=0, a=1, b=pi, k=1.5,
            distargs=None):
        assert N >= 0
        assert a > 0
        assert 0 <= b <= TWOPI
        assert k > 0
        # Sufficient statistics.
        self.N = N
        self.sum_sin_x = sum_sin_x
        self.sum_cos_x = sum_cos_x
        # Hyperparameters.
        self.a = a    # Prior concentration of mean parameter.
        self.b = b    # Prior mean of mean parameter.
        self.k = k    # Vonmises kappa.

    def incorporate(self, x):
        assert 0 <= x <= 2*pi
        self.N += 1.0
        self.sum_sin_x += sin(x)
        self.sum_cos_x += cos(x)

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        assert 0 <= x and x <= 2*pi
        self.N -= 1.0
        self.sum_sin_x -= sin(x)
        self.sum_cos_x -= cos(x)

    def logpdf(self, x):
        return Vonmises.calc_predictive_logp(x, self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)

    def logpdf_marginal(self):
        logp = Vonmises.calc_logpdf_marginal(self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)
        return logp

    def logpdf_singleton(self, x):
        return Vonmises.calc_predictive_logp(x, 0, 0, 0, self.a, self.b,
            self.k)

    def simulate(self):
        an, bn = Vonmises.posterior_hypers(self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)
        # if not 0 <= bn <= 2*pi:
        #     import ipdb; ipdb.set_trace()
        mu = np.random.vonmises(bn - pi, an) + pi
        x = np.random.vonmises(mu - pi, self.k) + pi
        assert 0 <= x <= 2*pi
        return x

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        assert 0 <= hypers['b'] and hypers['b'] <= TWOPI
        self.a = hypers['a']
        self.b = hypers['b']
        self.k = hypers['k']

    def get_hypers(self):
        return {
            'a' : self.a,
            'b' : self.b,
            'k' : self.k
        }

    def get_suffstats(self):
        return {
            'N': self.N,
            'sum_sin_x' : self.sum_sin_x,
            'sum_cos_x' : self.sum_cos_x
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = float(len(X))
        ssx = np.sum(np.sin(X))
        scx = np.sum(np.cos(X))
        k = Vonmises.estimate_kappa(N, ssx, scx)
        grids['a'] = gu.log_linspace(1./N, N, n_grid)
        grids['b'] = np.linspace(TWOPI/n_grid, TWOPI, n_grid)
        grids['k'] = np.linspace(k, N*k, n_grid)
        return grids

    @staticmethod
    def name():
        return 'vonmises'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_sin_x, sum_cos_x, a, b, k):
        assert 0 <= x and x <= 2*pi
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*pi
        assert k > 0
        if x < 0 or x > 2*pi:
            return float('-inf')
        an, _ = Vonmises.posterior_hypers(N, sum_sin_x, sum_cos_x, a, b, k)
        am, _ = Vonmises.posterior_hypers(N + 1., sum_sin_x + sin(x),
            sum_cos_x + cos(x), a, b, k)
        ZN = Vonmises.calc_log_Z(an)
        ZM = Vonmises.calc_log_Z(am)
        return - LOG2PI - gu.log_bessel_0(k) + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*pi
        assert k > 0
        an, _ = Vonmises.posterior_hypers(N, sum_sin_x, sum_cos_x, a, b, k)
        Z0 = Vonmises.calc_log_Z(a)
        ZN = Vonmises.calc_log_Z(an)
        lp = float(-N) * (LOG2PI + gu.log_bessel_0(k)) + ZN - Z0
        if np.isnan(lp) or np.isinf(lp):
            import ipdb; ipdb.set_trace()
        return lp

    @staticmethod
    def posterior_hypers(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*pi
        assert k > 0
        p_cos = k * sum_cos_x + a * cos(b)
        p_sin = k * sum_sin_x + a * sin(b)
        an = (p_cos**2.0 + p_sin**2.0)**.5
        bn = - atan2(p_cos, p_sin) + pi/2
        return an, bn

    @staticmethod
    def calc_log_Z(a):
        assert a > 0
        return gu.log_bessel_0(a)

    @staticmethod
    def estimate_kappa(N, ssx, scx):
        if N == 0:
            return 10.**-6
        elif N == 1:
            return 10*pi
        else:
            rbar2 = (ssx / N) ** 2. + (scx / N) ** 2.
            rbar = rbar2 ** .5
            kappa = rbar*(2. - rbar2) / (1. - rbar2)

        A_p = lambda k : bessel_1(k) / bessel_0(k)

        Apk = A_p(kappa)
        kappa_1 = kappa - (Apk - rbar)/(1. - Apk**2 - (1. / kappa) * Apk)
        Apk = A_p(kappa_1)
        kappa = kappa_1 - (Apk - rbar)/(1. - Apk**2 - (1. / kappa_1) * Apk)
        Apk = A_p(kappa)
        kappa_1 = kappa - (Apk - rbar)/(1. - Apk**2 - (1. / kappa) * Apk)
        Apk = A_p(kappa_1)
        kappa = kappa_1 - (Apk - rbar)/(1. - Apk**2 - (1. / kappa_1) * Apk)

        if np.isnan(kappa):
            return 10.**-6
        else:
            return np.abs(kappa)
