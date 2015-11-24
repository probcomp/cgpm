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

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0

        self.b = hypers['b']
        self.a = hypers['a']

    def transition_params(self):
        return

    def insert_element(self, x):
        self.N += 1.0
        self.sum_x += x

    def remove_element(self, x):
        self.N -= 1.0
        self.sum_x -= x

    def predictive_logp(self, x):
        return Exponential.calc_predictive_logp(x, self.N, self.sum_x, self.a,
            self.b)

    def marginal_logp(self):
        return Exponential.calc_marginal_logp(self.N, self.sum_x, self.a,
            self.b)

    def singleton_logp(self, x):
        return Exponential.calc_predictive_logp(x, 0, 0, self.a,
            self.b)

    def predictive_draw(self):
        an, bn = Exponential.posterior_update_parameters(self.N, self.sum_x,
            self.a, self.b)
        draw = lomax.rvs(an, loc=1-bn) - (1 - bn)
        return draw

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
    def calc_predictive_logp(x, N, sum_x, a, b):
        if x < 0:
            return float('-inf')
        an, bn = Exponential.posterior_update_parameters(N, sum_x, a, b)
        am, bm = Exponential.posterior_update_parameters(N+1, sum_x+x, a, b)

        ZN = Exponential.calc_log_Z(an, bn)
        ZM = Exponential.calc_log_Z(am, bm)

        return  ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, a, b):
        an, bn = Exponential.posterior_update_parameters(N, sum_x, a, b)

        Z0 = Exponential.calc_log_Z(a, b)
        ZN = Exponential.calc_log_Z(an, bn)

        return ZN - Z0

    @staticmethod
    def transition_hypers(clusters, hypers, grids):
        a = hypers['a']
        b = hypers['b']

        which_hypers = [0,1]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_a = Exponential.calc_a_conditional_logps(clusters, grids['a'], b)
                a_index = gu.log_pflip(lp_a)
                a = grids['a'][a_index]
            elif hyper == 1:
                lp_b = Exponential.calc_b_conditional_logps(clusters, grids['b'], a)
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
    def calc_log_Z(a, b):
        Z =  gammaln(a) - a*log(b)
        return Z

    @staticmethod
    def calc_a_conditional_logps(clusters, a_grid, b):
        lps = []
        for a in a_grid:
            lp = Exponential.calc_full_marginal_conditional(clusters, a, b)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_b_conditional_logps(clusters, b_grid, a):
        lps = []
        for b in b_grid:
            lp = Exponential.calc_full_marginal_conditional(clusters, a, b)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_full_marginal_conditional(clusters, a, b):
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            l = Exponential.calc_marginal_logp(N, sum_x, a, b)
            lp += l

        return lp

    @staticmethod
    def calc_full_marginal_conditional_h(clusters, hypers):
        lp = 0
        a = clusters[0].a
        b = clusters[0].b
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            l = Exponential.calc_marginal_logp(N, sum_x, a, b)
            lp += l

        return lp

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        x_min = 0
        x_max = max(X)
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
            N = clusters[k].N
            sum_x = clusters[k].sum_x
            for j in range(200):
                pdf[k, j] = np.exp(w + Exponential.calc_predictive_logp(Y[j], N,
                    sum_x, a, b))
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
