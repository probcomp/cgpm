from math import log

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import gammaln
from scipy.stats import gamma, expon

import gpmcc.utils.general as gu

class ExponentialUC(object):
    """ExponentialUC (count) data with gamma prior on mu.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'exponential_uc'

    def __init__(self, N=0, sum_x=0, a=2, b=2, mu=None, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- a: hyperparameter
        -- b: hyperparameter
        -- distargs: not used
        """
        assert a > 0
        assert b > 0
        self.N = N
        self.sum_x = sum_x
        self.a = a
        self.b = b

        if mu is None:
            self.mu = ExponentialUC.draw_params(self.a, self.b)

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0

        self.b = hypers['b']
        self.a = hypers['a']

    def transition_params(self):
        an, bn = ExponentialUC.posterior_update_parameters(self.N, self.sum_x,
            self.a, self.b)
        mu = self.draw_params(an, bn)
        self.mu = mu

    def insert_element(self, x):
        self.N += 1.0
        self.sum_x += x

    def remove_element(self, x):
        self.N -= 1.0
        self.sum_x -= x

    def predictive_logp(self, x):
        return ExponentialUC.calc_logp(x, self.mu)

    def marginal_logp(self):
        lp = ExponentialUC.calc_log_likelihood(self.N, self.sum_x, self.mu)
        return lp

    def singleton_logp(self, x):
        return ExponentialUC.calc_logp(x, self.mu)

    def predictive_draw(self):
        return expon.rvs(scale=1./self.mu)

    @staticmethod
    def calc_logp(x, mu):
        return expon.logpdf(x, scale=1./mu)

    @staticmethod
    def calc_log_likelihood(N, sum_x, mu):
        log_p =  N * log(mu) - mu * sum_x
        return log_p

    @staticmethod
    def draw_params(a, b):
        mu = gamma.rvs(a, scale=1./b)
        return mu

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['a'] = gu.log_linspace(1.0/float(len(X)), float(len(X)),
            n_grid)
        grids['b'] = gu.log_linspace(1.0/float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['a'] = np.random.choice(grids['a'])
        hypers['b'] = np.random.choice(grids['b'])

        return hypers

    @staticmethod
    def log_gamma(mu, a, b):
        return gamma.logpdf(mu, a, scale=1./b)

    @staticmethod
    def transition_hypers(clusters, hypers, grids):
        a = hypers['a']
        b = hypers['b']

        which_hypers = [0,1]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_a = ExponentialUC.calc_a_conditional_logps(clusters,
                    grids['a'], b)
                a_index = gu.log_pflip(lp_a)
                a = grids['a'][a_index]
            elif hyper == 1:
                lp_b = ExponentialUC.calc_b_conditional_logps(clusters,
                    grids['b'], a)
                b_index = gu.log_pflip(lp_b)
                b = grids['b'][b_index]
            else:
                raise ValueError("Invalid hyper.")

        hypers = dict()
        hypers['a'] = a
        hypers['b'] = b

        for cluster in clusters:
            cluster.set_hypers(hypers)

        return hypers

    @staticmethod
    def posterior_update_parameters(N, sum_x, a, b):
        an = a + N
        bn = b + sum_x
        return an, bn

    @staticmethod
    def calc_a_conditional_logps(clusters, a_grid, b):
        lps = []
        for a in a_grid:
            lp = 0
            for cluster in clusters:
                mu = cluster.mu
                lp += ExponentialUC.log_gamma(mu, a, b)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_b_conditional_logps(clusters, b_grid, a):
        lps = []
        for b in b_grid:
            lp = 0
            for cluster in clusters:
                mu = cluster.mu
                lp += ExponentialUC.log_gamma(mu, a, b)
            lps.append(lp)

        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        x_min = 0
        x_max = max(X) + 2
        Y = np.linspace(x_min, x_max, 200)
        K = len(clusters)
        pdf = np.zeros((K,200))
        denom = log(float(len(X)))

        a = clusters[0].a
        b = clusters[0].b

        nbins = min([len(X), 50])
        ax.hist(X, nbins, normed=True, color="black", alpha=.5,
            edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]
        for k in range(K):
            w = W[k]
            mu = clusters[k].mu
            for j in range(200):
                pdf[k, j] = np.exp(w + ExponentialUC.calc_logp(Y[j], mu))
            if k >= 8:
                color = "white"
                alpha = .3
            else:
                color = gu.colors()[k]
                alpha=.7
            ax.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)

        ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        ax.set_title('exponential')
        return ax
