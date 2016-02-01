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

import math
import csv
from math import log

import numpy as np
from scipy.special import i0 as bessel_0
from scipy.misc import logsumexp
from scipy.special import gammaln

colors = ['red', 'blue', 'green', 'magenta', 'orange', 'purple', 'brown',
    'black']

def curve_color(k):
    return (colors[k], .7) if k < len(colors) else ('white', .3)

def log_bessel_0(x):
    besa = bessel_0(x)
    # If bessel_0(a) is inf, then use the exponential approximation to
    # prevent numerical overflow.
    if math.isinf(besa):
        I0 = x - .5*log(2*math.pi*x)
    else:
        I0 = log(besa)
    return I0

def log_normalize(logp):
    """Normalizes a np array of log probabilites."""
    return logp - logsumexp(logp)

def logp_crp(N, Nk, alpha):
    """Returns the log normalized P(N,K|alpha), where N is the number of
    customers and K is the number of tables.
    """
    return gammaln(alpha) + len(Nk)*log(alpha) - gammaln(N+alpha) \
        + np.sum(gammaln(Nk))

def logp_crp_unorm(N, K, alpha):
    """Returns the log unnormalized P(N,K|alpha), where N is the number of
    customers and K is the number of tables. Use for effeciency to avoid
    computing terms that are not a function of alpha.
    """
    return gammaln(alpha) + K*log(alpha) - gammaln(N+alpha)

def log_pflip(logp):
    """Categorical draw from a vector logp of log probabilities."""
    if len(logp) == 1:
        return 0
    p = np.exp(log_normalize(logp))
    assert math.fabs(1.0-sum(p)) < 10.0**(-10.0)
    return pflip(p)

def pflip(p):
    """Categorical draw from a vector p of probabilities."""
    if len(p) == 1:
        return 0
    p = np.asarray(p).astype(float)
    p /= sum(p)
    if not math.fabs(1.0 - sum(p)) < 10.0**(-10.0):
        import ipdb;
        ipdb.set_trace()
    return np.random.choice(range(len(p)), size=1, p=p)[0]

def log_linspace(a, b, n):
    """linspace from a to b with n entries over log scale (mor entries at
    smaller values).
    """
    return np.exp(np.linspace(log(a), log(b), n))

def log_nCk(n, k):
    """log(nchoosek(n,k)) with overflow protection."""
    assert n >= 0
    assert k >= 0
    assert n >= k
    if n == 0 or k == 0 or n == k:
        return 0
    return log(n) + gammaln(n) - log(k) - gammaln(k) - log(n-k) - gammaln(n-k)

def line_quad(x,y):
    """Quadrature over x, where y = f(x). Uses triangle rule."""
    s = 0
    for i in range(1,len(x)):
        a = x[i-1]
        b = x[i]
        fa = y[i-1]
        fb = y[i]
        s += (b-a)*(fa+fb)/2
    return s

def simulate_crp(N, alpha):
    """Generates a random, N-length partition from the CRP with parameter
    alpha.
    """
    assert N > 0
    assert alpha > 0.0
    alpha = float(alpha)

    partition = np.zeros(N, dtype=int)
    Nk = [1]
    for i in xrange(1,N):
        K = len(Nk)
        ps = np.zeros(K+1)
        for k in xrange(K):
            # Get the number of people sitting at table k.
            ps[k] = float(Nk[k])
        ps[K] = alpha
        ps /= (float(i) - 1 + alpha)
        assignment = pflip(ps)
        if assignment == K:
            Nk.append(1)
        elif assignment < K:
            Nk[assignment] += 1
        else:
            raise ValueError("Invalid assignment: %i, max=%i" % (assignment, K))
        partition[i] = assignment

    assert max(partition)+1 == len(Nk)
    assert len(partition)==N
    assert sum(Nk) == N

    K = len(Nk)
    if K > 1:
        np.random.shuffle(partition)
    return np.array(partition), Nk, K

def kl_array(support, log_true, log_inferred, is_discrete):
    """
    Inputs:
    -- support: np array of support intervals.
    -- log_true: log pdf at support for the "true" distribution.
    -- log_inferred: log pdf at support for the distribution to test against the
    "true" distribution.
    -- is_discrete: is this a discrete variable True/False.

    Returns:
    - KL divergence.
    """
    # KL divergence formula, recall X and Y are log
    F = (log_true - log_inferred) * np.exp(log_true)
    if is_discrete:
        kld = np.sum(F)
    else:
        # trapezoidal quadrature
        intervals = np.diff(support)
        fs = F[:-1] + (np.diff(F) / 2.0)
        kld = np.sum(intervals*fs)
    return kld

def csv_to_list(filename):
    """Reads the csv filename into a list of lists."""
    T = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            T.append(row)
    return T

def csv_to_array(filename):
    """Reads the csv filename into a numpy array."""
    return np.asarray(csv_to_list(filename)).astype(float).T

def csv_to_data_and_colnames(filename):
    TC = csv_to_list(filename)
    colnames = list(TC[0])
    Y = np.genfromtxt(filename, delimiter=',', skip_header=1)
    if len(Y.shape) == 1:
        return [Y], colnames
    X = []
    for col in range(Y.shape[1]):
        X.append(Y[:,col])
    return np.asarray(X).T, colnames

def clean_data(X, cctypes):
    """Makes sure that discrete data columns are integer types."""
    is_discrete = {
        'normal'            : False,
        'bernoulli'         : False,  # 0. or 1.
        'categorical'       : True,
        'lognormal'         : False,
        'poisson'           : False,
        'exponential'       : False,
        'geometric'         : True,
        'vonmises'          : False,
    }

    for i in xrange(len(X)):
        if is_discrete[cctypes[i]]:
            X[i] = np.array(X[i], dtype=int)
    return X
