import math
from math import log

import numpy as np
import scipy

import matplotlib.pyplot as plt

import gpmcc.utils.general as gu
from gpmcc.dists.normal import Normal

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2*math.pi)

class NormalUC(Normal):
    """Normal data with normal prior on mean and gamma prior on precision.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'normal_uc'

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, mu=None, rho=None, m=0,
            r=1, s=1, nu=1, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- sum_x_sq: suffstat, sum(X^2)
        -- mu: parameter
        -- rho: parameter
        -- m: hyperparameter
        -- r: hyperparameter
        -- s: hyperparameter
        -- nu: hyperparameter
        -- distargs: not used
        """
        super(NormalUC, self).__init__(N=N, sum_x=sum_x, sum_x_sq=sum_x_sq,
            m=m, r=r, s=s, nu=nu, distargs=distargs)
        # Uncollapsed mean and precision parameters.
        self.mu, self.rho = mu, rho
        if mu is None or rho is None:
            self.transition_params()

    def transition_params(self):
        rn, nun, mn, sn = NormalUC.posterior_hypers(self.N, self.sum_x,
            self.sum_x_sq, self. m, self.r, self.s, self.nu)
        self.rho = np.random.gamma(nun/2., scale=2./sn)
        self.mu = np.random.normal(loc=mn, scale=1./(self.rho*rn)**.5)

    def predictive_logp(self, x):
        return NormalUC.calc_predictive_logp(x, self.mu, self.rho)

    def marginal_logp(self):
        lp = NormalUC.calc_log_likelihood(self.N, self.sum_x, self.sum_x_sq,
            self.rho, self.mu)
        return lp

    def singleton_logp(self, x):
        return NormalUC.calc_predictive_logp(x, self.mu, self.rho)

    def predictive_draw(self):
        return np.random.normal(self.mu, 1./self.rho**.5)

    @staticmethod
    def calc_predictive_logp(x, mu, rho):
        return scipy.stats.norm.logpdf(x, loc=mu, scale=1./rho**.5)

    @staticmethod
    def calc_log_likelihood(N, sum_x, sum_x_sq, rho, mu):
        return -(N / 2.) * LOG2PI + (N / 2.) * log(rho) - \
            .5 * (rho * (N * mu * mu - 2 * mu * sum_x + sum_x_sq))

    @staticmethod
    def calc_log_prior(mu, rho, m, r, s, nu):
        """Distribution of parameters (mu rho) ~ NG(m, r, s, nu)"""
        log_rho = scipy.stats.gamma.logpdf(rho, nu/2., scale=2./s)
        log_mu = scipy.stats.norm.logpdf(mu, loc=m, scale=1./(r*rho)**.5)
        return log_mu + log_rho

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(NormalUC.calc_log_prior(cluster.mu, cluster.rho,
                    **hypers) for cluster in clusters)
            lps.append(lp)
        return lps
