from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaln

import gpmcc.utils.general as gu

class Geometric(object):
    """Geometric (count) data with gamma prior on mu.
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

    def transition_params(self, prior=False):
        return

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
        # an, bn = Geometric.posterior_update_parameters(self.N, self.sum_x,
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
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['a'] = np.random.choice(grids['a'])
        hypers['b'] = np.random.choice(grids['b'])

        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, a, b):
        if float(x) != int(x) or x < 0:
            return float('-inf')
        an, bn = Geometric.posterior_update_parameters(N, sum_x, a, b)
        am, bm = Geometric.posterior_update_parameters(N+1, sum_x+x, a, b)

        ZN = Geometric.calc_log_Z(an, bn)
        ZM = Geometric.calc_log_Z(am, bm)

        return  ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, a, b):
        an, bn = Geometric.posterior_update_parameters(N, sum_x, a, b)

        Z0 = Geometric.calc_log_Z(a, b)
        ZN = Geometric.calc_log_Z(an, bn)

        return ZN - Z0

    @staticmethod
    def transition_hypers(clusters, hypers, grids):
        a = hypers['a']
        b = hypers['b']

        which_hypers = [0,1]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_a = Geometric.calc_a_conditional_logps(clusters, grids['a'], b)
                a_index = gu.log_pflip(lp_a)
                a = grids['a'][a_index]
            elif hyper == 1:
                lp_b = Geometric.calc_b_conditional_logps(clusters, grids['b'], a)
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
        Z =  betaln(a, b)
        return Z

    @staticmethod
    def calc_a_conditional_logps(clusters, a_grid, b):
        lps = []
        for a in a_grid:
            lp = Geometric.calc_full_marginal_conditional(clusters, a, b)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_b_conditional_logps(clusters, b_grid, a):
        lps = []
        for b in b_grid:
            lp = Geometric.calc_full_marginal_conditional(clusters, a, b)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_full_marginal_conditional(clusters, a, b):
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            l = Geometric.calc_marginal_logp(N, sum_x, a, b)
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
            l = Geometric.calc_marginal_logp(N, sum_x, a, b)
            lp += l

        return lp

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None):

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
            for j in xrange(nn):
                y = Y[j]
                pdf[k, j] = np.exp(w + Geometric.calc_predictive_logp(y, N,
                    sum_x, a, b))
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
        ax.set_title('geometric')
        return ax
