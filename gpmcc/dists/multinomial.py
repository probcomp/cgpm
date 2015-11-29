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
        assert float(distargs['k']) == int(distargs['k'])
        self.k = int(distargs['k'])
        if w is None:
            self.w = [0]*self.k
        else:
            assert self.k == len(w)
            self.w = w
        self.alpha = alpha
        self.N = N

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def transition_params(self, prior=False):
        return

    def insert_element(self, x):
        if not Multinomial.validate(x, self.k):
            raise ValueError('Invalid categorical observation inserted.')
        self.N += 1
        self.w[int(x)] += 1

    def remove_element(self, x):
        if not Multinomial.validate(x, self.k):
            raise ValueError('Invalid categorical observation removed.')
        self.N -= 1
        self.w[int(x)] -= 1

    def predictive_logp(self, x):
        return Multinomial.calc_predictive_logp(x, self.N, self.w, self.alpha)

    def marginal_logp(self):
        return Multinomial.calc_marginal_logp(self.N, self.w, self.alpha)

    def singleton_logp(self, x):
        return Multinomial.calc_predictive_logp(x, 0, [0]*self.k,
            self.alpha)

    def predictive_draw(self):
        return gu.pflip(self.w)

    @staticmethod
    def validate(x, K):
        assert int(x) == float(x)
        assert 0 <= x and x < K
        return int(x) == float(x) and 0 <= x and x < K

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
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
        if not Multinomial.validate(x, len(w)):
            return float('-inf')
        x = int(x)
        numer = log(alpha + w[x])
        denom = log( np.sum(w) + alpha * len(w))
        return numer - denom

    @staticmethod
    def calc_marginal_logp(N, w, alpha):
        K = len(w)
        A = K * alpha
        lg = sum(gammaln(w[k] + alpha) for k in xrange(K))
        return gammaln(A) - gammaln(A+N) + lg - K * gammaln(alpha)

    @staticmethod
    def transition_hypers(clusters, hypers, grids):
        # Do not need to extract any hypers since alpha is the only one.
        lp_alpha = Multinomial.calc_alpha_conditional_logps(clusters,
            grids['alpha'])
        alpha_index = gu.log_pflip(lp_alpha)
        hypers = dict()
        hypers['alpha'] = grids['alpha'][alpha_index]

        for cluster in clusters:
            cluster.set_hypers(hypers)

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
    def plot_dist(X, clusters, distargs=None, ax=None, hist=True):
        if ax is None:
            _, ax = plt.subplots()

        Y = range(int(distargs['k']))
        X_hist = np.array(gu.bincount(X,Y))
        X_hist = X_hist / float(len(X))

        K = len(clusters)
        pdf = np.zeros((K, int(distargs['k'])))
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
