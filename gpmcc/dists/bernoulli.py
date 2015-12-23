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
from scipy.special import betaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Bernoulli(DistributionGpm):
    """Bernoulli distribution with beta prior on bias theta.

    theta ~ Beta(alpha, beta)
    x ~ Bernoulli(theta)
    """

    def __init__(self, N=0, k=0, alpha=1, beta=1, distargs=None):
        assert alpha > 0
        assert beta > 0
        # Sufficient statistics.
        self.N = N
        self.k = k
        # Hyperparameter.
        self.alpha = alpha
        self.beta = beta

    def incorporate(self, x):
        assert x == 1.0 or x == 0.0
        self.N += 1
        self.k += x

    def unincorporate(self, x):
        assert x == 1. or x == 0.
        self.N -= 1
        self.k -= x

    def predictive_logp(self, x):
        return Bernoulli.calc_predictive_logp(x, self.N, self.k, self.alpha,
            self.beta)

    def marginal_logp(self):
        return Bernoulli.calc_marginal_logp(self.N, self.k, self.alpha,
            self.beta)

    def singleton_logp(self, x):
        return Bernoulli.calc_predictive_logp(x, 0, 0, self.alpha, self.beta)

    def simulate(self):
        if np.random.random() < self.alpha / (self.alpha + self.beta):
            return 1.
        else:
            return 0.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        assert hypers['beta'] > 0
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    def get_hypers(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta
        }

    def get_suffstats(self):
        return {
            'N' : self.N,
            'k' : self.k
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        grids['beta'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        X_hist = np.histogram(X,bins=2)[0]
        X_hist = X_hist/float(len(X))
        # Compute weighted pdfs
        Y = [0, 1]
        K = len(clusters)
        pdf = np.zeros((K, 2))
        ax.bar(Y, X_hist, color='black', alpha=1, edgecolor='none')
        W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
        if math.fabs(sum(np.exp(W)) -1.) > 10. ** (-10.):
            import ipdb; ipdb.set_trace()
        for k in xrange(K):
            pdf[k, :] = np.exp([W[k] + clusters[k].predictive_logp(y)
                    for y in Y])
            color, alpha = gu.curve_color(k)
            ax.bar(Y, pdf[k,:], color=color, edgecolor='none', alpha=alpha)
        # Plot the sum of pdfs.
        ax.bar(Y, np.sum(pdf, axis=0), color='none', edgecolor="red",
            linewidth=3)
        ax.set_xlim([-.1,1.9])
        ax.set_ylim([0,1.0])
        # Title
        ax.set_title(clusters[0].name())
        return ax

    @staticmethod
    def name():
        return 'bernoulli'

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, N, k, alpha, beta):
        if int(x) not in [0, 1]:
            return float('-inf')
        log_denom = log(N + alpha + beta)
        if x == 1.0:
            return log(k + alpha) - log_denom
        else:
            return log(N - k + beta) - log_denom

    @staticmethod
    def calc_marginal_logp(N, k, alpha, beta):
        return gu.log_nchoosek(N, k) + betaln(k + alpha, N - k + beta) \
            - betaln(alpha, beta)
