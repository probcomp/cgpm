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
        """ Optional arguments:
        -- N: number of data points
        -- w: weights
        -- alpha: dirichlet prior parameter
        -- distargs: dict with 'K' entry. K = len(w)
        """
        assert float(distargs['k']) == int(distargs['k'])
        self.k = int(distargs['k'])
        # Sufficient statistics.
        self.N = N
        if w is None:
            self.w = [0]*self.k
        else:
            assert self.k == len(w)
            self.w = w
        # Hyperparameter.
        self.alpha = alpha

    def transition_params(self, prior=False):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

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
        grids['alpha'] = gu.log_linspace(1./float(len(X)), float(len(X)),
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
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(Multinomial.calc_marginal_logp(cluster.N, cluster.w,
                **hypers) for cluster in clusters)
            lps.append(lp)
        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        Y = range(int(distargs['k']))
        X_hist = np.array(gu.bincount(X,Y))
        X_hist = X_hist / float(len(X))
        # Compute weighted pdfs
        K = len(clusters)
        pdf = np.zeros((K, int(distargs['k'])))
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
        ax.set_title(clusters[0].cctype)
        return ax
