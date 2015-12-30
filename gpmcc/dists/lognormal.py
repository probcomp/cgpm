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

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2.0*math.pi)


class Lognormal(DistributionGpm):
    """Log-normal (zero-bounded) distribution with normal prior on mean and
    gamma prior on precision. Collapsed.

    rho ~ Gamma(a, b)
    mu ~ Normal(m, t)
    x ~ Lognormal(mu, rho)
    """

    def __init__(self, N=0, sum_log_x=0, sum_log_x_sq=0, a=1, b=1, t=1,
            m=0, distargs=None):
        assert a > 0
        assert b > 0
        assert t > 0
        # Sufficient statistics.
        self.N = N
        self.sum_log_x_sq = sum_log_x_sq
        self.sum_log_x = sum_log_x
        # Hyperparameters.
        self.m = m
        self.a = a
        self.b = b
        self.t = t

    def incorporate(self, x):
        self.N += 1.0
        self.sum_log_x += log(x)
        self.sum_log_x_sq += log(x) * log(x)

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
        self.N -= 1.0
        self.sum_log_x -= log(x)
        self.sum_log_x_sq -= log(x) * log(x)

    def predictive_logp(self, x):
        return Lognormal.calc_predictive_logp(x, self.N, self.sum_log_x,
            self.sum_log_x_sq, self.a, self.b, self.t, self.m)

    def marginal_logp(self):
        return Lognormal.calc_marginal_logp(self.N, self.sum_log_x,
            self.sum_log_x_sq, self.a, self.b, self.t, self.m)

    def singleton_logp(self, x):
        return Lognormal.calc_predictive_logp(x, 0, 0, 0, self.a, self.b,
            self.t, self.m)

    def simulate(self):
        # XXX TODO
        raise NotImplementedError

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        assert hypers['t'] > 0
        self.a = hypers['a']
        self.b = hypers['b']
        self.t = hypers['t']
        self.m = hypers['m']

    def get_hypers(self):
        return {
            'a': self.a,
            'b': self.b,
            't': self.t,
            'm': self.m
        }

    def get_suffstats(self):
        return {
            'N': self.N,
            'sum_log_x': self.sum_log_x,
            'sum_log_x_sq': self.sum_log_x_sq
        }

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        ssqdev = np.var(X) * float(len(X));
        grids['a'] = gu.log_linspace(.1,float(len(X)), n_grid)
        grids['b'] = gu.log_linspace(.1,float(len(X)), n_grid)
        grids['t'] = gu.log_linspace(.1,float(len(X)), n_grid)
        grids['m'] = gu.log_linspace(.0001, max(X), n_grid)
        return grids

    @staticmethod
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        x_min = min(X)
        x_max = max(X)
        if Y is None:
            Y = np.linspace(x_min, x_max, 200)
        # Compute weighted pdfs
        K = len(clusters)
        pdf = np.zeros((K, len(Y)))
        W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
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
        return 'lognormal'

    @staticmethod
    def is_collapsed():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, sum_log_x, sum_log_x_sq, a, b, t, m):
        if x <= 0:
            return float('-inf')
        an, bn, tn, mn = Lognormal.posterior_hypers(N, sum_log_x,
            sum_log_x_sq, a, b, t, m)
        am, bm, tm, mm = Lognormal.posterior_hypers(N + 1,
            sum_log_x + log(x), sum_log_x_sq + log(x) * log(x), a, b, t, m)
        ZN = log(x) + Lognormal.calc_log_Z(an, bn, tn)
        ZM = Lognormal.calc_log_Z(am, bm, tm)
        return -0.5 * LOG2PI + ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_log_x, sum_log_x_sq, a, b, t, m):
        an, bn, tn, mn = Lognormal.posterior_hypers(N, sum_log_x,
            sum_log_x_sq, a, b, t, m)
        Z0 = sum_log_x + Lognormal.calc_log_Z(a, b, t)
        ZN = Lognormal.calc_log_Z(an, bn, tn)
        return -(float(N) / 2.) * LOG2PI + ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_log_x, sum_log_x_sq, a, b, t, m):
        tn = t + float(N)
        an = a + float(N)
        mn = (t * m + sum_log_x)/ tn
        bn = b + sum_log_x_sq + t * m * m - tn * mn * mn
        return an, bn, tn, mn

    @staticmethod
    def calc_log_Z(a, b, t):
        return ((a + 1.) / 2.) * LOG2 + .5 * LOGPI - .5 * log(t) - \
            (a / 2.) * log(b) + gammaln(a / 2.)
