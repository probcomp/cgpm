from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

import gpmcc.utils.general as gu

class Multinomial(object):
    """Multinomial with symmetric dirichlet prior on weights.
    Requires distargs to specify number of values (ex: distargs={'K':5})
    """

    cctype = 'multinomial'

    def __init__(self, N=0, w=None, alpha=1, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- w: weights
        -- alpha: dirichlet prior parameter
        -- distargs: dict with 'K' entry. K = len(w)
        """
        self.N = N
        self.K = distargs['K']
        if w is None:
            self.w = [0]*self.K
        else:
            assert self.K == len(w)
            self.w = w
        self.alpha = alpha

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def insert_element(self, x):
        assert np.issubdtype( type(x), np.integer)
        assert x >=0 and x < self.K
        self.N += 1
        self.w[x] += 1

    def remove_element(self, x):
        assert np.issubdtype( type(x), np.integer)
        assert x >=0 and x < self.K
        self.N -= 1
        self.w[x] -= 1

    def predictive_logp(self, x):
        assert np.issubdtype( type(x), np.integer)
        assert x >=0 and x < self.K
        return self.calc_predictive_logp(x, self.N, self.w, self.alpha)

    def singleton_logp(self, x):
        assert np.issubdtype( type(x), np.integer)
        assert x >=0 and x < self.K
        return self.calc_predictive_logp(x, 0, [0]*self.K, self.alpha)

    def marginal_logp(self):
        return self.calc_marginal_logp(self.N, self.w, self.alpha)

    def predictive_draw(self):
        return gu.pflip(self.w)

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        grids['alpha'] = gu.log_linspace(1.0 / float(len(X)), float(len(X)),
            n_grid)
        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['alpha'] = np.random.choice(grids['alpha'])
        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, w, alpha):
        numer = log(alpha + w[x])
        denom = log( np.sum(w) + alpha * len(w))
        return numer - denom

    @staticmethod
    def calc_marginal_logp(N, w, alpha):
        K = len(w)
        A = K*alpha
        lg = 0
        for k in range(K):
            lg += gammaln(w[k] + alpha)

        return gammaln(A) - gammaln(A+N) + lg - K * gammaln(alpha)


    @staticmethod
    def update_hypers(clusters, grids):
        # resample alpha
        lp_alpha = Multinomial.calc_alpha_conditional_logps(clusters,
            grids['alpha'])
        alpha_index = gu.log_pflip(lp_alpha)
        hypers = dict()
        hypers['alpha'] = grids['alpha'][alpha_index]

        return hypers

    @staticmethod
    def calc_alpha_conditional_logps(clusters, alpha_grid):
        lps = []
        for alpha in alpha_grid:
            lp = 0
            for cluster in clusters:
                N = cluster.N
                w = cluster.w
                lp += Multinomial.calc_marginal_logp(N, w, alpha)
            lps.append(lp)

        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        Y = range(distargs['K'])
        X_hist = np.array(gu.bincount(X,Y))
        X_hist = X_hist / float(len(X))

        K = len(clusters)
        pdf = np.zeros((K,distargs['K']))
        denom = log(float(len(X)))

        a = clusters[0].alpha

        ax.bar(Y, X_hist, color="black", alpha=1, edgecolor="none")
        W = [log(clusters[k].N) - denom for k in range(K)]
        for k in range(K):
            w = W[k]
            N = clusters[k].N
            ww = clusters[k].w
            for n in range(len(Y)):
                y = Y[n]
                pdf[k, n] = np.exp(w + Multinomial.calc_predictive_logp(y, N,
                    ww, a))
            ax.bar(Y, pdf[k,:], color="white", edgecolor="none", alpha=.5)

        ax.bar(Y, np.sum(pdf, axis=0), color='none', edgecolor="red",
            linewidth=1)
        # ax.ylim([0,1.0])
        ax.set_title('multinomial')
        return ax
