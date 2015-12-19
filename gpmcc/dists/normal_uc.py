import math
from math import log

import numpy as np
import scipy

import matplotlib.pyplot as plt
import pylab

import gpmcc.utils.general as gu

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2*math.pi)

class NormalUC(object):
    """Normal data with normal prior on mean and gamma prior on precision.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'normal_uc'

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, mu=None, rho=None, m=0, r=1,
            s=1, nu=1, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- sum_x_sq: suffstat, sum(X^2)
        -- m: hyperparameter
        -- r: hyperparameter
        -- s: hyperparameter
        -- nu: hyperparameter
        -- distargs: not used
        """
        assert s > 0.0
        assert r > 0.0
        assert nu > 0.0

        self.N = N
        self.sum_x = sum_x
        self.sum_x_sq = sum_x_sq

        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

        self.mu, self.rho = mu, rho
        if mu is None or rho is None:
            self.mu, self.rho = NormalUC.draw_params(m, r, s, nu)

    def set_hypers(self, hypers):
        assert hypers['s'] > 0.0
        assert hypers['r'] > 0.0
        assert hypers['nu'] > 0.0

        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def transition_params(self):
        rn, nun, mn, sn = NormalUC.posterior_update_parameters(self.N,
                self.sum_x, self.sum_x_sq, self. m, self.r, self.s, self.nu)
        mu, rho = self.draw_params(mn, rn, sn, nun)
        self.mu = mu
        self.rho = rho

    def insert_element(self, x):
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def remove_element(self, x):
        self.N -= 1.0
        self.sum_x -= x
        self.sum_x_sq -= x*x

    def predictive_logp(self, x):
        return self.calc_predictive_logp(x, self.mu, self.rho)

    def marginal_logp(self):
        lp = self.calc_log_likelihood(self.N, self.sum_x, self.sum_x_sq,
            self.rho, self.mu)
        return lp

    def singleton_logp(self, x):
        return NormalUC.calc_predictive_logp(x, self.mu, self.rho)

    def predictive_draw(self):
        return np.random.normal(self.mu, 1.0/self.rho**.5)

    @staticmethod
    def calc_predictive_logp(x, mu, rho):
        return scipy.stats.norm.logpdf(x, loc=mu, scale=1.0/rho**.5)

    @staticmethod
    def draw_params(m, r, s, nu):
        rho = np.random.gamma(nu/2.0, scale=2.0/s)
        mu = np.random.normal(loc=m, scale=1.0/(rho*r)**.5)
        return mu, rho

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        ssqdev = np.var(X) * float(len(X)) + 1
        grids['m'] = np.linspace(min(X),max(X)+5, n_grid)
        grids['s'] = gu.log_linspace(ssqdev/100.0, ssqdev, n_grid)
        grids['r'] = gu.log_linspace(1.0/(float(len(X))+1), float(len(X))+1,
            n_grid)
        grids['nu'] = gu.log_linspace(1.0,float(len(X)+1), n_grid) # df >= 1
        return grids

    @staticmethod
    def log_normal_gamma(mu, rho, m, r, s, nu):
        log_mu = scipy.stats.norm.logpdf(mu, loc=m, scale=1.0/(r*rho)**.5)
        log_rho = scipy.stats.gamma.logpdf(rho, nu/2.0, scale=2.0/s)
        return log_mu + log_rho

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['m'] = np.random.choice(grids['m'])
        hypers['s'] = np.random.choice(grids['s'])
        hypers['r'] = np.random.choice(grids['r'])
        hypers['nu'] = np.random.choice(grids['nu'])

        return hypers

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, rho, mu):
        log_p = -(N / 2.0) * LOG2PI + (N / 2.0) * log(rho) - \
            .5 * (rho * (N * mu * mu - 2 * mu * sum_x + sum_x_sq))
        return log_p

    @staticmethod
    def transition_hypers(clusters, hypers, grids):
        # resample hypers
        m = hypers['m']
        s = hypers['s']
        r = hypers['r']
        nu = hypers['nu']

        which_hypers = [0,1,2,3]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_m = NormalUC.calc_m_conditional_logps(clusters, grids['m'],
                    r, s, nu)
                m_index = gu.log_pflip(lp_m)
                m = grids['m'][m_index]
            elif hyper == 1:
                lp_s = NormalUC.calc_s_conditional_logps(clusters, grids['s'],
                    m, r, nu)
                s_index = gu.log_pflip(lp_s)
                s = grids['s'][s_index]
            elif hyper == 2:
                lp_r = NormalUC.calc_r_conditional_logps(clusters, grids['r'],
                    m, s, nu)
                r_index = gu.log_pflip(lp_r)
                r = grids['r'][r_index]
            elif hyper == 3:
                lp_nu = NormalUC.calc_nu_conditional_logps(clusters,
                    grids['nu'], m, r, s)
                nu_index = gu.log_pflip(lp_nu)
                nu = grids['nu'][nu_index]
            else:
                raise ValueError("Invalid hyper.")

        hypers = dict()
        hypers['m'] = m
        hypers['s'] = s
        hypers['r'] = r
        hypers['nu'] = nu

        for cluster in clusters:
            cluster.set_hypers(hypers)

        return hypers

    @staticmethod
    def posterior_update_parameters(N, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + float(N)
        nun = nu + float(N)
        mn = (r*m + sum_x)/rn
        sn = s + sum_x_sq + r*m*m - rn*mn*mn

        assert sn > 0

        return rn, nun, mn, sn

    @staticmethod
    def calc_m_conditional_logps(clusters, m_grid, r, s, nu):
        lps = []
        for m in m_grid:
            lp = 0
            for cluster in clusters:
                mu = cluster.mu
                rho = cluster.rho
                lp += NormalUC.log_normal_gamma(mu, rho, m, r, s, nu)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_r_conditional_logps(clusters, r_grid, m, s, nu):
        lps = []
        for r in r_grid:
            lp = 0
            for cluster in clusters:
                mu = cluster.mu
                rho = cluster.rho
                lp += NormalUC.log_normal_gamma(mu, rho, m, r, s, nu)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_s_conditional_logps(clusters, s_grid, m, r, nu):
        lps = []
        for s in s_grid:
            lp = 0
            for cluster in clusters:
                mu = cluster.mu
                rho = cluster.rho
                lp += NormalUC.log_normal_gamma(mu, rho, m, r, s, nu)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_nu_conditional_logps(clusters, nu_grid, m, r, s):
        lps = []
        for nu in nu_grid:
            lp = 0
            for cluster in clusters:
                mu = cluster.mu
                rho = cluster.rho
                lp += NormalUC.log_normal_gamma(mu, rho, m, r, s, nu)
            lps.append(lp)

        return lps


    @staticmethod
    def plot_dist(X, clusters, distargs=None, ax=None, Y=None, hist=True):
        if ax is None:
            _, ax = plt.subplots()

        x_min = min(X)
        x_max = max(X)
        if Y is None:
            Y = np.linspace(x_min, x_max, 200)
        K = len(clusters)
        pdf = np.zeros((K,200))
        denom = log(float(len(X)))

        W = [log(clusters[k].N) - denom for k in range(K)]
        for k in range(K):
            w = W[k]
            mu = clusters[k].mu
            rho = clusters[k].rho
            for n in range(200):
                y = Y[n]
                pdf[k, n] = np.exp(w + NormalUC.calc_predictive_logp(y, mu, rho))
            if k >= 8:
                color = "white"
                alpha=.3
            else:
                color = gu.colors()[k]
                alpha=.7
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

        ax.set_title('normal (uncollapsed)')
        return ax
