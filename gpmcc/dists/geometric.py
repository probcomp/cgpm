# -*- coding: utf-8 -*-

# The MIT License (MIT)

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
import matplotlib.pyplot as plt
from scipy.special import betaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Geometric(DistributionGpm):
    """Geometric distribution data with beta prior on mu. Collapsed.

    mu ~ Beta(a, b)
    x ~ Geometric(mu)
    """

    def __init__(self, N=0, sum_x=0, a=1, b=1, distargs=None):
        assert a > 0
        assert b > 0
        # Sufficient statistics.
        self.N = N
        self.sum_x = sum_x
        # Hyperparameters.
        self.a = a
        self.b = b

    def incorporate(self, x):
        self.N += 1.0
        self.sum_x += x

    def unincorporate(self, x):
        if self.N == 0:
            raise ValueError('Cannot unincorporate without observations.')
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

    def simulate(self):
        # XXX TODO
        raise NotImplementedError

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.b = hypers['b']
        self.a = hypers['a']

    def get_hypers(self):
        return {
            'a': self.a,
            'b': self.b,
        }

    def get_suffstats(self):
        return {
            'N': self.N,
            'sum_x': self.sum_x,
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(1, float(len(X)) / 2., n_grid)
        grids['b'] = gu.log_linspace(.1, float(len(X)) / 2., n_grid)
        return grids

    @staticmethod
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
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
        ax.set_title(clusters[0].name())
        return ax

    @staticmethod
    def name():
        return 'geometric'

    @staticmethod
    def is_collapsed():
        return True

    ##################
    # HELPER METHODS #
    ##################

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
