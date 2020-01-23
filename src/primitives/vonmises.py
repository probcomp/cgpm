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
from builtins import str
from past.utils import old_div
from math import atan2
from math import cos
from math import isinf
from math import isnan
from math import log
from math import pi
from math import sin

import numpy as np

from scipy.special import i0 as bessel_0
from scipy.special import i1 as bessel_1

from cgpm.primitives.distribution import DistributionGpm
from cgpm.utils import general as gu


class Vonmises(DistributionGpm):
    """Von Mises distribution on [0, 2pi] with Vonmises prior on the mean
    mu. The concentration k is fixed at construct time (defaults to 1.5).

    mu ~ Vonmises(mean=b, concentration=a)
    x ~ Vonmises(mean=mu, concentration=k)
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        DistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        # Sufficient statistics.
        self.N = 0
        self.sum_sin_x = 0
        self.sum_cos_x = 0
        # Hyperparameters.
        # Prior concentration of mean, mean of mean, and Vonmises kappa
        if hypers is None: hypers = {}
        self.a = hypers.get('a', 1.)
        self.b = hypers.get('b', pi)
        self.k = hypers.get('k', 1.5)
        assert self.a > 0
        assert 0 <= self.b <= 2*pi
        assert self.k > 0

    def incorporate(self, rowid, observation, inputs=None):
        DistributionGpm.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if not (0 <= x <= 2*pi):
            raise ValueError('Invalid Vonmises: %s' % str(x))
        self.N += 1
        self.sum_sin_x += sin(x)
        self.sum_cos_x += cos(x)
        self.data[rowid] = x

    def unincorporate(self, rowid):
        x = self.data.pop(rowid)
        self.N -= 1
        self.sum_sin_x -= sin(x)
        self.sum_cos_x -= cos(x)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionGpm.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if not (0 <= x <= 2*pi):
            return -float('inf')
        return Vonmises.calc_predictive_logp(
            x, self.N, self.sum_sin_x, self.sum_cos_x, self.a, self.b, self.k)

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionGpm.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        an, bn = Vonmises.posterior_hypers(
            self.N, self.sum_sin_x, self.sum_cos_x, self.a, self.b, self.k)
        # if not 0 <= bn <= 2*pi:
        #     import ipdb; ipdb.set_trace()
        mu = self.rng.vonmises(bn-pi, an) + pi
        x = self.rng.vonmises(mu-pi, self.k) + pi
        assert 0 <= x <= 2*pi
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return Vonmises.calc_logpdf_marginal(
            self.N, self.sum_sin_x, self.sum_cos_x, self.a, self.b, self.k)

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        assert 0 <= hypers['b'] and hypers['b'] <= 2*pi
        self.a = hypers['a']
        self.b = hypers['b']
        self.k = hypers['k']

    def get_hypers(self):
        return {'a' : self.a, 'b' : self.b, 'k' : self.k}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {'N': self.N, 'sum_sin_x' : self.sum_sin_x,
            'sum_cos_x' : self.sum_cos_x}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = float(len(X))
        ssx = np.sum(np.sin(X))
        scx = np.sum(np.cos(X))
        k = Vonmises.estimate_kappa(N, ssx, scx)
        grids['a'] = gu.log_linspace(1./N, N, n_grid)
        grids['b'] = np.linspace(old_div(2*pi,n_grid), 2*pi, n_grid)
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
    def calc_predictive_logp(x, N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert k > 0
        an, _ = Vonmises.posterior_hypers(N, sum_sin_x, sum_cos_x, a, b, k)
        am, _ = Vonmises.posterior_hypers(N + 1., sum_sin_x + sin(x),
            sum_cos_x + cos(x), a, b, k)
        ZN = Vonmises.calc_log_Z(an)
        ZM = Vonmises.calc_log_Z(am)
        return - np.log(2*pi) - log_bessel_0(k) + ZM - ZN

    @staticmethod
    def calc_logpdf_marginal(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*pi
        assert k > 0
        an, _ = Vonmises.posterior_hypers(N, sum_sin_x, sum_cos_x, a, b, k)
        Z0 = Vonmises.calc_log_Z(a)
        ZN = Vonmises.calc_log_Z(an)
        lp = float(-N) * (np.log(2*pi) + log_bessel_0(k)) + ZN - Z0
        if isnan(lp) or isinf(lp):
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
        bn = - atan2(p_cos, p_sin) + old_div(pi,2)
        return an, bn

    @staticmethod
    def calc_log_Z(a):
        assert a > 0
        return log_bessel_0(a)

    @staticmethod
    def estimate_kappa(N, ssx, scx):
        if N == 0:
            return 10.**-6
        elif N == 1:
            return 10*pi
        else:
            rbar2 = (old_div(ssx, N)) ** 2. + (old_div(scx, N)) ** 2.
            rbar = rbar2 ** .5
            kappa = old_div(rbar*(2. - rbar2), (1. - rbar2))

        A_p = lambda k : old_div(bessel_1(k), bessel_0(k))

        Apk = A_p(kappa)
        kappa_1 = kappa - old_div((Apk - rbar),(1. - Apk**2 - (1. / kappa) * Apk))
        Apk = A_p(kappa_1)
        kappa = kappa_1 - old_div((Apk - rbar),(1. - Apk**2 - (1. / kappa_1) * Apk))
        Apk = A_p(kappa)
        kappa_1 = kappa - old_div((Apk - rbar),(1. - Apk**2 - (1. / kappa) * Apk))
        Apk = A_p(kappa_1)
        kappa = kappa_1 - old_div((Apk - rbar),(1. - Apk**2 - (1. / kappa_1) * Apk))

        if isnan(kappa):
            return 10.**-6
        else:
            return abs(kappa)

def log_bessel_0(x):
    besa = bessel_0(x)
    # If bessel_0(a) is inf, then use the exponential approximation to
    # prevent numerical overflow.
    if isinf(besa):
        I0 = x - .5*log(2*pi*x)
    else:
        I0 = log(besa)
    return I0
