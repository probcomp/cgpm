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

from scipy.stats import gamma, expon

from gpmcc.dists.exponential import Exponential

class ExponentialUC(Exponential):
    """Exponential distribution with gamma prior on mu. Uncollapsed.

    mu ~ Gamma(a, b)
    x ~ Exponential(mu)
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
        return expon.rvs(scale=1./self.mu)

    def transition_params(self):
        an, bn = Exponential.posterior_hypers(self.N, self.sum_x, self.a,
            self.b)
        self.mu = gamma.rvs(an, scale=1./bn)

    @staticmethod
    def name():
        return 'exponential_uc'

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def calc_predictive_logp(x, mu):
        return expon.logpdf(x, scale=1./mu)

    @staticmethod
    def calc_log_likelihood(N, sum_x, mu):
        return  N * log(mu) - mu * sum_x

    @staticmethod
    def calc_log_prior(mu, a, b):
        return gamma.logpdf(mu, a, scale=1./b)
