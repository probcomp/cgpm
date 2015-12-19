import math
import warnings
from math import log

import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2*math.pi)

class Normal(object):
    """Normal data with normal prior on mean and gamma prior on precision.
    Does not require additional argumets (distargs=None).
    """

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

    def set_hypers(self, hypers):
        assert hypers['s'] > 0.0
        assert hypers['r'] > 0.0
        assert hypers['nu'] > 0.0

        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def transition_params(self):
        return

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
        rn, nun, mn, sn = Normal.posterior_update_parameters(self.N,
            self.sum_x, self.sum_x_sq, self.m, self.r, self.s, self.nu)
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
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['m'] = np.random.choice(grids['m'])
        hypers['s'] = np.random.choice(grids['s'])
        hypers['r'] = np.random.choice(grids['r'])
        hypers['nu'] = np.random.choice(grids['nu'])

        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_update_parameters(N, sum_x, sum_x_sq,
            m, r, s, nu)
        rm, num, mm, sm = Normal.posterior_update_parameters(
            N+1, sum_x+x, sum_x_sq+x*x, m, r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        ZM = Normal.calc_log_Z(rm, sm, num)
        return -.5*LOG2PI + ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_update_parameters(N, sum_x, sum_x_sq,
            m, r, s, nu)
        Z0 = Normal.calc_log_Z(r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)
        return - (float(N) / 2.0) * LOG2PI + ZN - Z0

    @staticmethod
    def transition_hypers(clusters, hypers, grids):
        hypers = hypers.copy()
        targets = hypers.keys()
        np.random.shuffle(targets)

        for target in hypers.keys():
            lp = Normal.calc_hyper_logps(clusters, grids[target], hypers,
                target)
            proposal = gu.log_pflip(lp)
            hypers[target] = grids[target][proposal]

        for cluster in clusters:
            cluster.set_hypers(hypers)

        return hypers

    @staticmethod
    def posterior_update_parameters(N, sum_x, sum_x_sq, m, r, s, nu):
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
        Z = ((nu + 1.) / 2.) * LOG2 + .5 * LOGPI - .5 * log(r) \
                - (nu / 2.) * log(s) + gammaln(nu/2.0)
        return Z

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = Normal.calc_clusters_marginal_logp(clusters, **hypers)
            lps.append(lp)
        return lps

    @staticmethod
    def calc_clusters_marginal_logp(clusters, m, r, s, nu):
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            sum_x_sq = cluster.sum_x_sq
            l = Normal.calc_marginal_logp(N, sum_x, sum_x_sq, m, r, s, nu)
            lp += l
        return lp

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
        # Compute weighted pdfs
        K = len(clusters)
        pdf = np.zeros((K, 200))
        denom = log(float(len(X)))
        W = [log(clusters[k].N) - denom for k in range(K)]
        for k in range(K):
            for n in range(200):
                y = Y[n]
                pdf[k, n] = np.exp(W[k] + clusters[k].predictive_logp(y))
            if k >= 8:
                color = "white"
                alpha = .3
            else:
                color = gu.colors()[k]
                alpha = .7
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
        # Title.
        ax.set_title(clusters[0].cctype)
        return ax
