# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class cgpm.primitives.normal.Normal uses derivations from both

    http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf (Section 3)

The two sources use different parameterizations. This suite ensures that the
conversions of these parameterizations performed in Normal produce consistent
numerical results between the two sources.

In particular, truncating the Normal requires computing the logcdf to normalize
the probability, which is a Student T derived in Murphy but not Teh.
"""

import itertools as it

import numpy as np
from numpy import log, pi, sqrt
from scipy.special import gammaln
from scipy.stats import t

from cgpm.primitives.normal import Normal
from cgpm.utils.general import gen_rng


# Prepare some functions for use in the test.
def teh_posterior(m, r, s, nu, x):
    """Eq 10 Teh."""
    N = len(x)
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x**2)
    return Normal.posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu)


def murphy_posterior(a, b, k, mu, x):
    """Eqs 85 to 89 Murphy."""
    n = len(x)
    xbar = np.mean(x)
    kn = k + n
    an = a + n/2.
    mun = (k*mu+n*xbar)/(k+n)
    bn = b + .5*np.sum((x-xbar)**2) + k*n*(xbar-mu)**2 / (2*(k+n))
    return an, bn, kn, mun


def murphy_posterior_predictive(an1, an, bn1, bn, kn1, kn):
    """Eq 99 Murphy."""
    return gammaln(an1) - gammaln(an) + an*log(bn) - \
        (an1)*log(bn1) + 1/2.*(log(kn) - log(kn1)) - 1/2.*log(2*pi)


# Test suite.

def test_agreement():
    # Hyperparmaeters in Teh notation.
    all_m = map(float, (1., 7., .43, 1.2))
    all_r = map(float, (2., 18., 3., 11.))
    all_s = map(float, (2., 6., 15., 55.))
    all_nu = map(float, (4., .6, 14., 8.))

    # Dataset
    rng = gen_rng(0)
    x1 = rng.normal(10, 3, size=100)
    x2 = rng.normal(-3, 7, size=100)

    for (m, r, s, nu), x in \
            it.product(zip(all_m, all_r, all_s, all_nu), [x1,x2]):
        # Murphy hypers in terms of Teh.
        a = nu/2.
        b = s/2.
        k = r
        mu = m

        # Test equality of posterior hypers.
        mn, rn, sn, nun = teh_posterior(m, r, s, nu, x)
        an, bn, kn, mun = murphy_posterior(a, b, k, mu, x)
        assert np.allclose(an, nun/2, atol=1e-5)
        assert np.allclose(bn, sn/2, atol=1e-5)
        assert np.allclose(kn, rn, atol=1e-5)
        assert np.allclose(mun, mn, atol=1e-5)

        # Test posterior predictive agree with each other, and Student T.
        for xtest in np.linspace(1.1, 80.8, 14.1):
            # Murphy exact, Eq 99.
            an1, bn1, kn1, mun1 = murphy_posterior(
                a, b, k, mu, np.append(x, xtest))
            logprob_murphy = murphy_posterior_predictive(
                an1, an, bn1, bn, kn1, kn)

            # Student T Murphy, Eq 100.
            scalesq = bn*(kn+1)/(an*kn)
            logprob_t_murphy = t.logpdf(
                xtest, 2*an, loc=mun, scale=sqrt(scalesq))

            # Teh exact using Murphy Eq 99.
            mn1, rn1, sn1, nun1 = teh_posterior(
                m, r, s, nu, np.append(x, xtest))
            logprob_teh = murphy_posterior_predictive(
                nun1/2., nun/2, sn1/2., sn/2, rn1, rn)

            # Posterior predictive from Normal DistributionGpm.
            logprob_nignormal = Normal.calc_predictive_logp(
                xtest, len(x), sum(x), np.sum(x**2), m, r, s, nu)

            # Student T Teh using Murphy Eq 100.
            scalesq = sn/2.*(rn+1)/(nun/2.*rn)
            logprob_t_teh = t.logpdf(
                xtest, 2*nun/2., loc=mn, scale=sqrt(scalesq))

            # Aggregate all values and test their equality.
            values = [logprob_murphy, logprob_teh, logprob_t_murphy,
                logprob_t_teh, logprob_nignormal]
            for v in values:
                assert np.allclose(v, values[0], atol=1e-2)
