# -*- coding: utf-8 -*-

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math
from math import log
from math import sin
from math import cos

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0 as bessel_0
from scipy.special import i1 as bessel_1

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

TWOPI = 2 * math.pi
LOG2PI = log(2 * math.pi)

class Vonmises(DistributionGpm):
    """Von Mises distribution, assuming fixed concentration k."""

    cctype = 'vonmises'

    def __init__(self, N=0, sum_sin_x=0, sum_cos_x=0, a=1, b=math.pi, k=1.5,
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
        assert 0 <= x <= 2*math.pi
        self.N += 1.0
        self.sum_sin_x += sin(x)
        self.sum_cos_x += cos(x)

    def unincorporate(self, x):
        assert 0 <= x and x <= 2*math.pi
        self.N -= 1.0
        self.sum_sin_x -= sin(x)
        self.sum_cos_x -= cos(x)

    def predictive_logp(self, x):
        return Vonmises.calc_predictive_logp(x, self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)

    def marginal_logp(self):
        logp = Vonmises.calc_marginal_logp(self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)
        return logp

    def singleton_logp(self, x):
        return Vonmises.calc_predictive_logp(x, 0, 0, 0, self.a, self.b,
            self.k)

    def simulate(self):
        fn = lambda x: np.exp(self.predictive_logp(x))
        lower_bound = 0.0
        delta = 2*math.pi/10000
        return gu.inversion_sampling(fn, lower_bound, delta)

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
        grids['a'] = gu.log_linspace(1/N, N, n_grid)
        grids['b'] = np.linspace(TWOPI/n_grid, TWOPI, n_grid)
        grids['k'] = np.linspace(k, N*k, n_grid)
        return grids

    @staticmethod
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        x_min = 0
        x_max = 2 * math.pi
        Y = np.linspace(x_min, x_max, 200)
        # Compute weighted pdfs
        K = len(clusters)
        pdf = np.zeros((K, 200))
        W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
        if math.fabs(sum(np.exp(W)) -1.0) > 10.0 ** (-10.0):
            import ipdb; ipdb.set_trace()
        for k in xrange(K):
            pdf[k, :] = np.exp([W[k] + clusters[k].predictive_logp(y)
                    for y in Y])
            color, alpha = gu.curve_color(k)
            ax.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)
        # Plot the sum of pdfs.
        ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        # Plot the samples.
        if hist:
            nbins = min([len(X)/5, 50])
            ax.hist(X, nbins, normed=True, color='black', alpha=.5,
                edgecolor='none')
        else:
            y_max = ax.get_ylim()[1]
            for x in X:
                ax.vlines(x, 0, y_max/10., linewidth=1)
        # Title.
        ax.set_title(clusters[0].name())
        return ax

    @staticmethod
    def name():
        return 'vonmises'

    @staticmethod
    def is_collapsed():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_sin_x, sum_cos_x, a, b, k):
        assert 0 <= x and x <= 2 * math.pi
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2 * math.pi
        assert k > 0
        if x < 0 or x > math.pi * 2.:
            return float('-inf')
        an, bn = Vonmises.posterior_hypers(N, sum_sin_x, sum_cos_x, a, b, k)
        am, bm = Vonmises.posterior_hypers(N + 1., sum_sin_x + sin(x),
            sum_cos_x + cos(x), a, b, k)
        ZN = Vonmises.calc_log_Z(an)
        ZM = Vonmises.calc_log_Z(am)
        return - LOG2PI - gu.log_bessel_0(k) + ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2 * math.pi
        assert k > 0
        an, bn = Vonmises.posterior_hypers(N, sum_sin_x,
            sum_cos_x, a, b, k)
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
        assert 0 <= b and b <= 2*math.pi
        assert k > 0
        p_cos = k * sum_cos_x + a * math.cos(b)
        p_sin = k * sum_sin_x + a * math.sin(b)
        an = (p_cos**2.0 + p_sin**2.0)**.5
        bn = -1 # not used. There may be a time...
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
            return 10 * math.pi
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
