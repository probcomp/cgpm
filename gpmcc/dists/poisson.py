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

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0

        self.b = hypers['b']
        self.a = hypers['a']

    def transition_params(self, prior=False):
        return

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
            lp = Poisson.calc_clusters_marginal_logp(clusters, **hypers)
            lps.append(lp)
        return lps

    @staticmethod
    def calc_clusters_marginal_logp(clusters, a, b):
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            sum_log_fact_x = cluster.sum_log_fact_x
            l = Poisson.calc_marginal_logp(N, sum_x, sum_log_fact_x, a, b)
            lp += l
        return lp

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        if ax is None:
            _, ax = plt.subplots()

        x_max = max(X)
        Y = range(int(x_max)+1)
        nn = len(Y)
        K = len(clusters)
        pdf = np.zeros((K,nn))
        denom = log(float(len(X)))

        a = clusters[0].a
        b = clusters[0].b

        toplt = np.array(gu.bincount(X,Y))/float(len(X))
        ax.bar(Y, toplt, color="gray", edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]
        for k in xrange(K):
            w = W[k]
            N = clusters[k].N
            sum_x = clusters[k].sum_x
            sum_log_fact_x = clusters[k].sum_log_fact_x
            for n in xrange(nn):
                y = Y[n]
                pdf[k, n] = np.exp(w + Poisson.calc_predictive_logp(y, N, sum_x,
                    sum_log_fact_x, a, b))
            if k >= 8:
                color = "white"
                alpha = .3
            else:
                color = gu.colors()[k]
                alpha = .7
            ax.bar(Y, pdf[k,:], color=color, edgecolor='none', alpha=alpha)

        ax.bar(Y, np.sum(pdf, axis=0), color='none', edgecolor='black',
            linewidth=3)
        # print integral for debugging (should never be greater that 1)
        # print gu.line_quad(Y, np.sum(pdf,axis=0))
        ax.set_xlim([0, x_max+1])
        ax.set_title('poisson')
        return ax
