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
        self.N = N
        self.sum_log_x = sum_log_x
        self.sum_minus_log_x = sum_minus_log_x

        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        self.strength, self.balance = strength, balance
        if strength is None or balance is None:
            self.strength, self.balance = BetaUC.draw_params(mu, alpha, beta)

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

    def resample_params(self):
        n_samples = 25

        log_pdf_lambda_str = lambda strength : BetaUC.calc_logp(self.N,
            self.sum_log_x, self.sum_minus_log_x, strength, self.balance) \
            + BetaUC.calc_log_prior(strength, self.balance, self.mu,
                self.alpha, self.beta)
        self.strength = su.mh_sample(self.strength, log_pdf_lambda_str,
            .5, [0.0,float('Inf')], burn=n_samples)

        log_pdf_lambda_bal = lambda balance : BetaUC.calc_logp(self.N,
            self.sum_log_x, self.sum_minus_log_x, self.strength, balance) \
            + BetaUC.calc_log_prior(self.strength, balance, self.mu,
                self.alpha, self.beta)
        self.balance = su.mh_sample(self.balance, log_pdf_lambda_bal,
            .25, [0,1], burn=n_samples)

    def predictive_logp(self, x):
        return BetaUC.calc_singleton_logp(x, self.strength, self.balance)

    def singleton_logp(self, x):
        return BetaUC.calc_singleton_logp(x, self.strength, self.balance)

    def marginal_logp(self):
        lp = BetaUC.calc_logp(self.N, self.sum_log_x, self.sum_minus_log_x,
            self.strength, self.balance)
        lp += BetaUC.calc_log_prior(self.strength, self.balance, self.mu,
            self.alpha, self.beta)
        return lp

    def predictive_draw(self):
        alpha = self.strength*self.balance
        beta = self.strength*(1.0-self.balance)
        return np.random.beta(alpha, beta)

    def set_hypers(self, hypers):
        self.mu = hypers['mu']
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    @staticmethod
    def calc_singleton_logp(x, strength, balance):
        assert strength > 0 and balance > 0 and balance < 1

        alpha = strength*balance
        beta = strength*(1.0-balance)
        lp = scipy.stats.beta.logpdf(x, alpha, beta)

        assert not np.isnan(lp)
        return lp

    @staticmethod
    def calc_log_prior(strength, balance, mu, alpha, beta):
        assert strength > 0 and balance > 0 and balance < 1

        lp = 0
        lp += scipy.stats.expon.logpdf(strength, scale=mu)
        lp += scipy.stats.beta.logpdf(balance, alpha, beta)
        return lp

    @staticmethod
    def calc_logp(N, sum_log_x, sum_minus_log_x, strength, balance):
        assert( strength > 0 and balance > 0 and balance < 1)

        alpha = strength*balance
        beta = strength*(1.0-balance)
        lp = 0
        lp -= N*scipy.special.betaln(alpha, beta)
        lp += (alpha-1.0)*sum_log_x
        lp += (beta-1.0)*sum_minus_log_x

        assert( not np.isnan(lp) )
        return lp

    @staticmethod
    def draw_params(mu, alpha, beta):
        strength = np.random.exponential(scale=mu)
        balance = np.random.beta(alpha, beta)

        assert strength > 0 and balance > 0 and balance < 1
        return strength, balance

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        N = float(len(X))
        Sx = np.sum(X)
        Mx = np.sum(1-X)
        grids = {
            'mu' : gu.log_linspace(1/N, N, n_grid),
            'alpha' : gu.log_linspace(Sx/N, Sx, n_grid),
            'beta' : gu.log_linspace(Mx/N, Mx, n_grid),
        }
        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = {
            'mu' : np.random.choice(grids['mu']),
            'alpha' : np.random.choice(grids['alpha']),
            'beta' : np.random.choice(grids['beta']),
        }
        return hypers

    @staticmethod
    def calc_prior_conditionals(clusters, mu, alpha, beta):
        lp = 0
        for cluster in clusters:
            strength = cluster.strength
            balance = cluster.balance
            lp += BetaUC.calc_log_prior(strength, balance, mu, alpha, beta)
        return lp

    @staticmethod
    def calc_mu_conditional_logps(clusters, mu_grid, alpha, beta):
        lps = []
        for mu in mu_grid:
            lps.append(BetaUC.calc_prior_conditionals(clusters, mu, alpha,
                beta))
        return lps

    @staticmethod
    def calc_alpha_conditional_logps(clusters, alpha_grid, mu, beta):
        lps = []
        for alpha in alpha_grid:
            lps.append(BetaUC.calc_prior_conditionals(clusters, mu, alpha,
                beta))
        return lps

    @staticmethod
    def calc_beta_conditional_logps(clusters, beta_grid, mu, alpha):
        lps = []
        for beta in beta_grid:
            lps.append(BetaUC.calc_prior_conditionals(clusters, mu, alpha,
                beta))
        return lps

    @staticmethod
    def resample_hypers(clusters, grids):
        # resample hypers
        mu = clusters[0].mu
        alpha = clusters[0].alpha
        beta = clusters[0].beta

        which_hypers = [0,1,2]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_mu = BetaUC.calc_mu_conditional_logps(clusters, grids['mu'],
                    alpha, beta)
                mu_index = gu.log_pflip(lp_mu)
                mu = grids['mu'][mu_index]
            elif hyper == 1:
                lp_alpha = BetaUC.calc_alpha_conditional_logps(clusters,
                    grids['alpha'], mu, beta)
                alpha_index = gu.log_pflip(lp_alpha)
                alpha = grids['alpha'][alpha_index]
            elif hyper == 2:
                lp_beta = BetaUC.calc_beta_conditional_logps(clusters,
                    grids['beta'], mu, alpha)
                beta_index = gu.log_pflip(lp_beta)
                beta = grids['beta'][beta_index]
            else:
                raise ValueError("Invalid hyper.")

        hypers = dict()
        hypers['mu'] = mu
        hypers['alpha'] = alpha
        hypers['beta'] = beta

        return hypers

    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        N = 100
        Y = np.linspace(0.01, .99, N)
        K = len(clusters)
        pdf = np.zeros((K,N))
        denom = log(float(len(X)))

        nbins = min([len(X)/5, 50])

        ax.hist(X, nbins, normed=True, color="black", alpha=.5,
            edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]
        for k in range(K):
            w = W[k]
            strength = clusters[k].strength
            balance = clusters[k].balance
            for n in range(N):
                y = Y[n]
                pdf[k, n] = np.exp(w + BetaUC.calc_singleton_logp(y,
                    strength, balance))
            if k >= 8:
                color = "white"
                alpha=.3
            else:
                color = gu.colors()[k]
                alpha=.7
            ax.plot(Y, pdf[k,:],color=color, linewidth=5, alpha=alpha)

        ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        ax.set_title('beta (uncollapsed)')
        return ax
