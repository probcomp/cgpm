from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import lomax

import gpmcc.utils.general as gu

class Exponential(object):
    """Exponential (count) data with gamma prior on mu.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'exponential'

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

    def remove_element(self, x):
        self.N -= 1.0
        self.sum_x -= x

    def predictive_logp(self, x):
        return Exponential.calc_predictive_logp(x, self.N, self.sum_x,
            self.a, self.b)

    def marginal_logp(self):
        return Exponential.calc_marginal_logp(self.N, self.sum_x, self.a,
            self.b)

    def singleton_logp(self, x):
        return Exponential.calc_predictive_logp(x, 0, 0, self.a,
            self.b)

    def predictive_draw(self):
        an, bn = Exponential.posterior_hypers(self.N,
            self.sum_x, self.a, self.b)
        draw = lomax.rvs(an, loc=1-bn) - (1 - bn)
        return draw

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        grids['b'] = gu.log_linspace(1./float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, a, b):
        if x < 0:
            return float('-inf')
        an,bn = Exponential.posterior_hypers(N, sum_x, a, b)
        am,bm = Exponential.posterior_hypers(N+1, sum_x+x, a, b)
        ZN = Exponential.calc_log_Z(an, bn)
        ZM = Exponential.calc_log_Z(am, bm)
        return  ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, a, b):
        an, bn = Exponential.posterior_hypers(N, sum_x, a, b)
        Z0 = Exponential.calc_log_Z(a, b)
        ZN = Exponential.calc_log_Z(an, bn)
        return ZN - Z0

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + N
        bn = b + sum_x
        return an, bn

    @staticmethod
    def calc_log_Z(a, b):
        Z =  gammaln(a) - a*log(b)
        return Z

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(Exponential.calc_marginal_logp(cluster.N,
                cluster.sum_x, **hypers) for cluster in clusters)
            lps.append(lp)
        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        x_min = 0
        x_max = max(X)
        if Y is None:
            Y = np.linspace(x_min, x_max, 200)
        # Compute weighted pdfs.
        K = len(clusters)
        pdf = np.zeros((K, len(Y)))
        denom = log(float(len(X)))
        W = [log(clusters[k].N) - denom for k in xrange(K)]
        for k in range(K):
            pdf[k, :] = np.exp([W[k] + clusters[k].predictive_logp(y)
                    for y in Y])
            color, alpha = gu.curve_color(k)
            ax.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)
        # Plot the sum of pdfs.
        ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        # Plot the samples.
        if hist:
            nbins = min([len(X), 50])
            ax.hist(X, nbins, normed=True, color="black", alpha=.5,
                edgecolor="none")
        else:
            y_max = ax.get_ylim()[1]
            for x in X:
                ax.vlines(x, 0, y_max/float(10), linewidth=1)
        ax.set_title(clusters[0].cctype)
        return ax
