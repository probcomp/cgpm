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

from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaln

import gpmcc.utils.general as gu

class Geometric(object):
    """Geometric distribution data with beta prior on mu."""

    cctype = 'geometric'

    def __init__(self, N=0, sum_x=0, a=1, b=1, distargs=None):
        """Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- a: hyperparameter
        -- b: hyperparameter
        -- distargs: not used
        """
        assert a > 0
        assert b > 0
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def transition_params(self, prior=False):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.b = hypers['b']
        self.a = hypers['a']

    def insert_element(self, x):
        self.N += 1.0
        self.sum_x += x

    def remove_element(self, x):
        self.N -= 1.0
        self.sum_x -= x

    def predictive_logp(self, x):
        return Geometric.calc_predictive_logp(x, self.N, self.sum_x, self.a,
            self.b)

    def marginal_logp(self):
        return Geometric.calc_marginal_logp(self.N, self.sum_x, self.a,
            self.b)

    def singleton_logp(self, x):
        return Geometric.calc_predictive_logp(x, 0, 0, self.a, self.b)

    def predictive_draw(self):
        # an, bn = Geometric.posterior_hypers(self.N, self.sum_x,
            # self.a, self.b)
        # XXX Fix.
        # draw = np.random.negative_binomial(an, bn/(bn+1.0))
        return 1
        # fn = lambda x: np.exp(self.predictive_logp(x))
        # lower_bound = 0
        # delta = 1
        # return utils.inversion_sampling(fn, lower_bound, delta)

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(1, float(len(X)) / 2., n_grid)
        grids['b'] = gu.log_linspace(.1, float(len(X)) / 2., n_grid)
        return grids

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, a, b):
        if float(x) != int(x) or x < 0:
            return float('-inf')
        an, bn = Geometric.posterior_hypers(N, sum_x, a, b)
        am, bm = Geometric.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Geometric.calc_log_Z(an, bn)
        ZM = Geometric.calc_log_Z(am, bm)
        return  ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, a, b):
        an, bn = Geometric.posterior_hypers(N, sum_x, a, b)
        Z0 = Geometric.calc_log_Z(a, b)
        ZN = Geometric.calc_log_Z(an, bn)
        return ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + N
        bn = b + sum_x
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  betaln(a, b)
        return Z

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(Geometric.calc_marginal_logp(cluster.N, cluster.sum_x,
                **hypers) for cluster in clusters)
            lps.append(lp)
        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        x_max = max(X)
        Y = range(int(x_max)+1)
        # Compute weighted pdfs
        K = len(clusters)
        pdf = np.zeros((K,len(Y)))
        toplt = np.array(gu.bincount(X,Y))/float(len(X))
        ax.bar(Y, toplt, color='gray', edgecolor='none')
        W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
        for k in xrange(K):
            pdf[k, :] = np.exp([W[k] + clusters[k].predictive_logp(y)
                    for y in Y])
            color, alpha = gu.curve_color(k)
            ax.bar(Y, pdf[k,:], color=color, edgecolor='none', alpha=alpha)
        # Plot the sum of pdfs.
        ax.bar(Y, np.sum(pdf, axis=0), color='none', edgecolor='black',
            linewidth=3)
        ax.set_xlim([0, x_max+1])
        # Title.
        ax.set_title(clusters[0].cctype)
        return ax
