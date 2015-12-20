from math import log

import numpy as np
import matplotlib.pyplot as plt
import scipy

import gpmcc.utils.general as gu
import gpmcc.utils.sampling as su

class BetaUC(object):
    """Beta distribution parameterized in terms of strength, s and balance b.

        theta ~ beta(s*b, s*(1-b)),
        s ~ exp(mu),
        b ~ beta(alpha, beta).

    Does not require additional argumets (distargs=None).
    """

    cctype = 'beta_uc'

    def __init__(self, N=0, sum_log_x=0, sum_minus_log_x=0, strength=None,
            balance=None, mu=1, alpha=.5, beta=.5, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_log_x: suffstat, sum(log(X))
        -- sum_minus_log_x: suffstat, sum(log(1-X))
        -- strength: higher strength -> lower variance
        -- balance: analogous to mean
        -- mu: hyperparam, exponential distribution parameter for strength prior
        -- alpha: hyperparam, beta distribution parameter for balance
        -- beta: hyperparam, beta distribution parameter for balance
        -- distargs: not used
        """
        # Sufficient statistics.
        self.N = N
        self.sum_log_x = sum_log_x
        self.sum_minus_log_x = sum_minus_log_x
        # Hyperparameters.
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        # Parameters.
        self.strength, self.balance = strength, balance
        if strength is None or balance is None:
            self.strength, self.balance = BetaUC.draw_params(mu, alpha,
                beta)
            self.strength = np.random.exponential(scale=mu)
            self.balance = np.random.beta(alpha, beta)
            assert self.strength > 0 and 0 < self.balance < 1

    def transition_params(self):
        n_samples = 25
        # Transition strength.
        log_pdf_lambda_str = lambda strength :\
            BetaUC.calc_log_likelihood(self.N, self.sum_log_x,
                self.sum_minus_log_x, strength, self.balance) \
            + BetaUC.calc_log_prior(strength, self.balance, self.mu,
                self.alpha, self.beta)
        self.strength = su.mh_sample(self.strength, log_pdf_lambda_str,
            .5, [0.0,float('Inf')], burn=n_samples)
        # Transition balance.
        log_pdf_lambda_bal = lambda balance : \
            BetaUC.calc_log_likelihood(self.N, self.sum_log_x,
                self.sum_minus_log_x, self.strength, balance) \
            + BetaUC.calc_log_prior(self.strength, balance, self.mu,
                self.alpha, self.beta)
        self.balance = su.mh_sample(self.balance, log_pdf_lambda_bal,
            .25, [0,1], burn=n_samples)

    def set_hypers(self, hypers):
        self.mu = hypers['mu']
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    def insert_element(self, x):
        assert x > 0 and x < 1
        self.N += 1.0
        self.sum_log_x += log(x)
        self.sum_minus_log_x += log(1.0-x)

    def remove_element(self, x):
        assert x > 0 and x < 1
        self.N -= 1.0
        if self.N <= 0:
            self.sum_log_x = 0
            self.sum_minus_log_x = 0
        else:
            self.sum_log_x -= log(x)
            self.sum_minus_log_x -= log(1.0-x)

    def predictive_logp(self, x):
        return BetaUC.calc_predictive_logp(x, self.strength, self.balance)

    def singleton_logp(self, x):
        return BetaUC.calc_predictive_logp(x, self.strength, self.balance)

    def marginal_logp(self):
        lp = BetaUC.calc_log_likelihood(self.N, self.sum_log_x,
            self.sum_minus_log_x, self.strength, self.balance)
        lp += BetaUC.calc_log_prior(self.strength, self.balance, self.mu,
            self.alpha, self.beta)
        return lp

    @staticmethod
    def calc_predictive_logp(x, strength, balance):
        assert strength > 0 and balance > 0 and balance < 1
        if not 0 < x < 1:
            return float('-inf')
        alpha = strength * balance
        beta = strength * (1.0-balance)
        lp = scipy.stats.beta.logpdf(x, alpha, beta)
        assert not np.isnan(lp)
        return lp

    @staticmethod
    def calc_log_prior(strength, balance, mu, alpha, beta):
        assert strength > 0 and balance > 0 and balance < 1
        log_strength = scipy.stats.expon.logpdf(strength, scale=mu)
        log_balance = scipy.stats.beta.logpdf(balance, alpha, beta)
        return log_strength + log_balance

    @staticmethod
    def calc_log_likelihood(N, sum_log_x, sum_minus_log_x, strength,
            balance):
        assert strength > 0 and balance > 0 and balance < 1
        alpha = strength * balance
        beta = strength * (1. - balance)
        lp = 0
        lp -= N*scipy.special.betaln(alpha, beta)
        lp += (alpha - 1.) * sum_log_x
        lp += (beta - 1.) * sum_minus_log_x
        assert not np.isnan(lp)
        return lp

    @staticmethod
    def draw_params(mu, alpha, beta):
        strength = np.random.exponential(scale=mu)
        balance = np.random.beta(alpha, beta)
        assert strength > 0 and balance > 0 and balance < 1
        return strength, balance

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = float(len(X))
        Sx = np.sum(X)
        Mx = np.sum(1-X)
        grids['mu'] = gu.log_linspace(1/N, N, n_grid)
        grids['alpha'] = gu.log_linspace(Sx/N, Sx, n_grid)
        grids['beta'] = gu.log_linspace(Mx/N, Mx, n_grid)
        return grids

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(BetaUC.calc_log_prior(cluster.strength,
                cluster.balance, **hypers) for cluster in clusters)
            lps.append(lp)
        return lps

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        # Create a new axis?
        if ax is None:
            _, ax = plt.subplots()
        # Set up x axis.
        if Y is None:
            Y = np.linspace(.01, .99, 100)
        # Compute weighted pdfs.
        K = len(clusters)
        pdf = np.zeros((K, len(Y)))
        W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
        for k in xrange(K):
            pdf[k, :] = np.exp([W[k] + clusters[k].predictive_logp(y)
                    for y in Y])
            color, alpha = gu.curve_color(k)
            ax.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)
        # Plot the sum of pdfs.
        ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        # Plot the samples.
        if hist:
            nbins = min([len(X)/5, 50])
            ax.hist(X, nbins, normed=True, color='black', alpha=.5,
                edgecolor='none')
        else:
            y_max = ax.get_ylim()[1]
            for x in X:
                ax.vlines(x, 0, y_max/10., linewidth=1)
        # Title.
        ax.set_title(clusters[0].cctype)
        return ax
