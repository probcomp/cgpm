from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu

class Poisson(object):
    """Poisson (count) data with gamma prior on mu.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'poisson'

    def __init__(self, N=0, sum_x=0, sum_log_fact_x=0, a=1, b=1, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- sum_x_log_fact_x: suffstat, sum(log(X!))
        -- a: hyperparameter
        -- b: hyperparameter
        -- distargs: not used
        """
        assert a > 0
        assert b > 0
        self.N = N
        self.sum_x = sum_x
        self.sum_log_fact_x = sum_log_fact_x
        self.a = a
        self.b = b

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.b = hypers['b']
        self.a = hypers['a']

    def insert_element(self, x):
        self.N += 1.0
        self.sum_x += x
        self.sum_log_fact_x += gammaln(x+1)

    def remove_element(self, x):
        self.N -= 1.0
        self.sum_x -= x
        self.sum_log_fact_x -= gammaln(x+1)

    def predictive_logp(self, x):
        return Poisson.calc_predictive_logp(x, self.N, self.sum_x,
            self.sum_log_fact_x, self.a, self.b)

    def marginal_logp(self):
        return Poisson.calc_marginal_logp(self.N, self.sum_x,
            self.sum_log_fact_x, self.a, self.b)

    def singleton_logp(self, x):
        return Poisson.calc_predictive_logp(x, 0, 0, 0, self.a,
            self.b)

    def predictive_draw(self):
        an, bn = Poisson.posterior_update_parameters(self.N, self.sum_x,
            self.a, self.b)
        draw = np.random.negative_binomial(an, bn/(bn+1.0))
        return draw

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        # only use integers for a so we can nicely draw from a negative binomial
        # in predictive_draw
        grids['a'] = np.unique(np.round(np.linspace(1, len(X), n_grid)))
        grids['b'] = gu.log_linspace(.1, float(len(X)), n_grid)
        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['a'] = np.random.choice(grids['a'])
        hypers['b'] = np.random.choice(grids['b'])
        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, sum_log_fact_x, a, b):
        if float(x) != x or x < 0:
            return float('-inf')
        an, bn = Poisson.posterior_update_parameters(N, sum_x, a, b)
        am, bm = Poisson.posterior_update_parameters(N+1, sum_x+x, a, b)
        ZN = Poisson.calc_log_Z(an, bn)
        ZM = Poisson.calc_log_Z(am, bm)
        return  ZM - ZN - gammaln(x+1)

    @staticmethod
    def calc_marginal_logp(N, sum_x, sum_log_fact_x, a, b):
        an, bn = Poisson.posterior_update_parameters(N, sum_x, a, b)
        Z0 = Poisson.calc_log_Z(a, b)
        ZN = Poisson.calc_log_Z(an, bn)
        return ZN - Z0 - sum_log_fact_x

    @staticmethod
    def posterior_update_parameters(N, sum_x, a, b):
        an = a + sum_x
        bn = b + N
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  gammaln(a)-a*log(b)
        return Z

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(Poisson.calc_marginal_logp(cluster.N, cluster.sum_x,
                cluster.sum_log_fact_x, **hypers) for cluster in clusters)
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
        pdf = np.zeros((K, len(Y)))
        toplt = np.array(gu.bincount(X,Y)) / float(len(X))
        ax.bar(Y, toplt, color='gray', edgecolor='none')
        W = [log(clusters[k].N) - log(float(len(X))) for k in range(K)]
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
