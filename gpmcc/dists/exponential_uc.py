# -*- coding: utf-8 -*-

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
