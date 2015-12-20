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
import warnings
from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2*math.pi)

class Normal(object):
    """Normal distribution with normal prior on mean and gamma prior on
    precision."""

    cctype = 'normal'

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, m=0, r=1, s=1, nu=1,
            distargs=None):
        """Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- sum_x_sq: suffstat, sum(X^2)
        -- m: hyperparameter
        -- r: hyperparameter
        -- s: hyperparameter
        -- nu: hyperparameter
        -- distargs: not used
        """
        assert s > 0.
        assert r > 0.
        assert nu > 0.
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        self.sum_x_sq = sum_x_sq
        # Hyper parameters.
        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['s'] > 0.
        assert hypers['r'] > 0.
        assert hypers['nu'] > 0.
        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def insert_element(self, x):
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def remove_element(self, x):
        self.N -= 1.0
        if self.N == 0:
            self.sum_x = 0.0
            self.sum_x_sq = 0.0
        else:
            self.sum_x -= x
            self.sum_x_sq -= x*x

    def predictive_logp(self, x):
        return Normal.calc_predictive_logp(x, self.N, self.sum_x,
            self.sum_x_sq, self.m, self.r, self.s, self.nu)

    def marginal_logp(self):
        return Normal.calc_marginal_logp(self.N, self.sum_x, self.sum_x_sq,
            self.m, self.r, self.s, self.nu)

    def singleton_logp(self, x):
        return Normal.calc_predictive_logp(x, 0, 0, 0, self.m, self.r,
            self.s, self.nu)

    def predictive_draw(self):
        rn, nun, mn, sn = Normal.posterior_hypers(self.N, self.sum_x,
            self.sum_x_sq, self.m, self.r, self.s, self.nu)
        coeff = ( ((sn / 2.) * (rn + 1.)) / ((nun / 2.) * rn) ) ** .5
        draw = np.random.standard_t(nun) * coeff + mn
        return draw

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
    def calc_predictive_logp(x, N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_hypers(N, sum_x, sum_x_sq, m, r,
            s, nu)
        rm, num, mm, sm = Normal.posterior_hypers(N+1, sum_x+x,
            sum_x_sq+x*x, m, r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        ZM = Normal.calc_log_Z(rm, sm, num)
        return -.5*LOG2PI + ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_hypers(N, sum_x, sum_x_sq, m, r,
            s, nu)
        Z0 = Normal.calc_log_Z(r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        return - (float(N) / 2.0) * LOG2PI + ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + float(N)
        nun = nu + float(N)
        mn = (r*m + sum_x)/rn
        sn = s + sum_x_sq + r*m*m - rn*mn*mn
        if sn == 0:
            warnings.warn('Posterior_update_parameters: sn truncated.')
            sn = s
        return rn, nun, mn, sn

    @staticmethod
    def calc_log_Z(r, s, nu):
        return ((nu + 1.) / 2.) * LOG2 + .5 * LOGPI - .5 * log(r) \
                - (nu / 2.) * log(s) + gammaln(nu/2.0)

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(Normal.calc_marginal_logp(cluster.N, cluster.sum_x,
                cluster.sum_x_sq, **hypers) for cluster in clusters)
            lps.append(lp)
        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        x_min = min(X)
        x_max = max(X)
        if Y is None:
            Y = np.linspace(x_min, x_max, 200)
        # Compute weighted pdfs.
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
            nbins = min([len(X), 50])
            ax.hist(X, nbins, normed=True, color='black', alpha=.5,
                edgecolor='none')
        else:
            y_max = ax.get_ylim()[1]
            for x in X:
                ax.vlines(x, 0, y_max/10., linewidth=1)
        # Title.
        ax.set_title(clusters[0].cctype)
        return ax
