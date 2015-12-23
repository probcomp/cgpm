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
from scipy.special import gammaln

import gpmcc.utils.general as gu
from gpmcc.dists.distribution import DistributionGpm

class Multinomial(DistributionGpm):
    """Multinomial distribution with symmetric dirichlet prior on
    category weight vector v.

    k := distarg
    v ~ Symmetric-Dirichlet(alpha/k)
    x ~ Multinomial(v)
    """

    def __init__(self, N=0, counts=None, alpha=1, distargs=None):
        # Number of categories.
        assert float(distargs['k']) == int(distargs['k'])
        self.k = int(distargs['k'])
        # Sufficient statistics.
        self.N = N
        if counts is None:
            self.counts = [0]*self.k
        else:
            assert self.k == len(counts)
            self.counts = counts
        # Hyperparameter.
        self.alpha = alpha

    def incorporate(self, x):
        if not Multinomial.validate(x, self.k):
            raise ValueError('Invalid categorical observation inserted.')
        self.N += 1
        self.counts[int(x)] += 1

    def unincorporate(self, x):
        if not Multinomial.validate(x, self.k):
            raise ValueError('Invalid categorical observation removed.')
        self.N -= 1
        self.counts[int(x)] -= 1

    def predictive_logp(self, x):
        return Multinomial.calc_predictive_logp(x, self.N, self.counts,
            self.alpha)

    def marginal_logp(self):
        return Multinomial.calc_marginal_logp(self.N, self.counts,
            self.alpha)

    def singleton_logp(self, x):
        return Multinomial.calc_predictive_logp(x, 0, [0]*self.k,
            self.alpha)

    def simulate(self):
        return gu.pflip(self.counts)

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def get_hypers(self):
        return {
            'alpha': self.alpha,
        }

    def get_suffstats(self):
        return {
            'N' : self.N,
            'counts' : self.counts
        }

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        Y = range(int(clusters[0].k))
        X_hist = np.array(gu.bincount(X,Y))
        X_hist = X_hist / float(len(X))
        # Compute weighted pdfs
        K = len(clusters)
        pdf = np.zeros((K, int(clusters[0].k)))
        ax.bar(Y, X_hist, color='black', alpha=1, edgecolor='none')
        W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
        for k in xrange(K):
            pdf[k, :] = np.exp([W[k] + clusters[k].predictive_logp(y)
                    for y in Y])
            color, alpha = gu.curve_color(k)
            ax.bar(Y, pdf[k,:], color=color, edgecolor='none', alpha=alpha)
        # Plot the sum of pdfs.
        ax.bar(Y, np.sum(pdf, axis=0), color='none', edgecolor="red",
            linewidth=1)
        # ax.ylim([0,1.0])
        # Title
        ax.set_title(clusters[0].name())
        return ax

    @staticmethod
    def name():
        return 'multinomial'

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def validate(x, K):
        assert int(x) == float(x)
        assert 0 <= x and x < K
        return int(x) == float(x) and 0 <= x and x < K

    @staticmethod
    def calc_predictive_logp(x, N, counts, alpha):
        if not Multinomial.validate(x, len(counts)):
            return float('-inf')
        x = int(x)
        numer = log(alpha + counts[x])
        denom = log(np.sum(counts) + alpha * len(counts))
        return numer - denom

    @staticmethod
    def calc_marginal_logp(N, counts, alpha):
        K = len(counts)
        A = K * alpha
        lg = sum(gammaln(counts[k] + alpha) for k in xrange(K))
        return gammaln(A) - gammaln(A+N) + lg - K * gammaln(alpha)
