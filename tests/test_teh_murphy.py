# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
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

"""
The class gpmcc.dists.normal.Normal uses derivations from both

    http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf (Section 3)

The two sources use different parameterizations. This suite ensures that the
conversions of these parameterizations performed in Normal produce consistent
numerical results between the two sources.

In particular, truncating the Normal requires computing the logcdf to normalize
the probability, which is a Student T derived in Murphy but not Teh.
"""

import numpy as np
from numpy import log, pi, sqrt
from scipy.special import gammaln
from scipy.stats import norm, t

from gpmcc.dists.normal import Normal

# Posterior hypers.
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

# Teh variables.
m = 1.
r = 2.
s = 2.
nu = 4.

# Murphy variables.
a = nu/2.
b = s/2.
k = r
mu = m

# Test equality of posterior hypers.
x = norm.rvs(loc=10, scale=3, size=100)
mn, rn, sn, nun = teh_posterior(m, r, s, nu, x)
an, bn, kn, mun = murphy_posterior(a, b, k, mu, x)

print np.allclose(an, nun/2)
print np.allclose(bn, sn/2)
print np.allclose(kn, rn)
print np.allclose(mun, mn)

# Test posterior predictive agree with each other, and Student T.
xtest_all = np.linspace(1.1, 80.8, 14.1)

for xtest in xtest_all:
    # Murphy exact, Eq 99.
    an1, bn1, kn1, mun1 = murphy_posterior(a, b, k, mu, np.append(x, xtest))
    logprob_murphy = gammaln(an1) - gammaln(an) + an*log(bn) - (an1)*log(bn1) \
        + 1/2.*(log(kn) - log(kn1)) - mu/2.*log(2*pi)

    # Student T Murphy, Eq 100.
    scalesq = bn*(kn+1)/(an*kn)
    logprob_t_murphy = t.logpdf(xtest, 2*an, loc=mun, scale=sqrt(scalesq))

    # Teh exact using Murphy Eq 99.
    mn1, rn1, sn1, nun1 = teh_posterior(m, r, s, nu, np.append(x, xtest))
    logprob_teh = gammaln(nun1/2.) - gammaln(nun/2.) + nun/2.*log(sn/2.) \
        - (nun1/2.)*log(sn1/2.) + 1/2.*(log(rn) - log(rn1)) - m/2.*log(2*pi)

    # Student T Teh using Murphy Eq 100.
    scalesq = sn/2.*(rn+1)/(nun/2.*rn)
    logprob_t_teh = t.logpdf(xtest, 2*nun/2., loc=mn, scale=sqrt(scalesq))

    values = [logprob_murphy, logprob_teh, logprob_t_murphy, logprob_t_teh]
    print np.allclose(values[0], values)
