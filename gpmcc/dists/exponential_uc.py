from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon

import gpmcc.utils.general as gu
from gpmcc.dists.exponential import Exponential

class ExponentialUC(Exponential):
    """ExponentialUC (count) data with gamma prior on mu.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'exponential_uc'

    def __init__(self, N=0, sum_x=0, a=2, b=2, mu=None, distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- a: hyperparameter
        -- b: hyperparameter
        -- distargs: not used
        """
        super(ExponentialUC, self).__init__(N=N, sum_x=sum_x, a=a, b=b,
            distargs=distargs)
        # Uncollapsed mean parameter.
        if mu is None:
            self.mu = ExponentialUC.draw_params(self.a, self.b)

    def transition_params(self):
        an, bn = ExponentialUC.posterior_update_parameters(self.N,
            self.sum_x, self.a, self.b)
        mu = self.draw_params(an, bn)
        self.mu = mu

    def predictive_logp(self, x):
        return ExponentialUC.calc_predictive_logp(x, self.mu)

    def marginal_logp(self):
        lp = ExponentialUC.calc_log_likelihood(self.N, self.sum_x, self.mu)
        return lp

    def singleton_logp(self, x):
        return ExponentialUC.calc_predictive_logp(x, self.mu)

    def predictive_draw(self):
        return expon.rvs(scale=1./self.mu)

    @staticmethod
    def calc_predictive_logp(x, mu):
        return expon.logpdf(x, scale=1./mu)

    @staticmethod
    def calc_log_likelihood(N, sum_x, mu):
        return  N * log(mu) - mu * sum_x

    @staticmethod
    def calc_log_prior(mu, a, b):
        return gamma.logpdf(mu, a, scale=1./b)

    @staticmethod
    def calc_hyper_logps(clusters, grid, hypers, target):
        lps = []
        for g in grid:
            hypers[target] = g
            lp = sum(ExponentialUC.calc_log_prior(cluster.mu, **hypers)
                for cluster in clusters)
            lps.append(lp)
        return lps

    @staticmethod
    def draw_params(a, b):
        return gamma.rvs(a, scale=1./b)
