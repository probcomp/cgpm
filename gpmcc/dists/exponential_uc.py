# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

from math import log

import numpy as np
from scipy.stats import invgamma, expon

from gpmcc.dists.exponential import Exponential

class ExponentialUC(Exponential):
    """Exponential distribution with gamma prior on mu, the mean.
    Uncollapsed.

    mu ~ InvGamma(a, b)
    x ~ Exponential(mu)
    http://seor.gmu.edu/~klaskey/SYST664/Bayes_Unit3.pdf
    """

    def __init__(self, N=0, sum_x=0, a=2, b=2, mu=None, distargs=None):
        # Invoke parent.
        super(ExponentialUC, self).__init__(N=N, sum_x=sum_x, a=a, b=b,
            distargs=distargs)
        # Uncollapsed mean parameter.
        if mu is None:
            self.transition_params()

    def predictive_logp(self, x):
        return ExponentialUC.calc_predictive_logp(x, self.mu)

    def marginal_logp(self):
        data_logp = ExponentialUC.calc_log_likelihood(self.N, self.sum_x,
            self.mu)
        prior_logp = ExponentialUC.calc_log_prior(self.mu, self.a, self.b)
        return data_logp + prior_logp

    def singleton_logp(self, x):
        return ExponentialUC.calc_predictive_logp(x, self.mu)

    def simulate(self):
        return expon.rvs(scale=self.mu)

    def transition_params(self):
        an, bn = Exponential.posterior_hypers(self.N, self.sum_x, self.a,
            self.b)
        self.mu = invgamma.rvs(an, scale=bn)
        if np.isinf(self.mu):
            import ipdb; ipdb.set_trace()

    @staticmethod
    def name():
        return 'exponential_uc'

    @staticmethod
    def is_collapsed():
        return False

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def posterior_hypers(N, sum_x, a, b):
        an = a + N
        bn = 1. / (1./b + sum_x)
        return an, bn

    @staticmethod
    def calc_predictive_logp(x, mu):
        return expon.logpdf(x, scale=mu)

    @staticmethod
    def calc_log_likelihood(N, sum_x, mu):
        return  - N * log(mu) - sum_x / mu

    @staticmethod
    def calc_log_prior(mu, a, b):
        if np.isinf(mu):
            import ipdb; ipdb.set_trace()
        return invgamma.logpdf(mu, a, scale=b)
