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

colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown',
    'black']

def curve_color(k):
    return (colors[k], .7) if k < len(colors) else ('white', .3)

def log_bessel_0(x):
    besa = bessel_0(x)
    # if bessel_0(a) is inf, then use the eponential approximation to
    # prevent numericala overflow
    if math.isinf(besa):
        I0 = x - .5*log(2*math.pi*x)
    else:
        I0 = log(besa)
    return I0

def log_normalize(logp):
    """Normalizes a np array of log probabilites."""
    return logp - logsumexp(logp)

def lcrp(N, Nk, alpha):
    """Returns the log probability under crp of the count vector Nk given
    concentration parameter alpha. N is the total number of entries.
    """
    N = float(N)
    k = float(len(Nk)) # number of classes
    l = np.sum(gammaln(Nk)) + k * log(alpha) + gammaln(alpha) - \
        gammaln(N + alpha)
    return l

def unorm_lcrp_post(alpha, N, K, log_prior_fun):
    """Returns the log of unnormalized P(alpha|N,K).

    Arguments:
    -- alpha: crp alpha parameter. float greater than 0
    -- N: number of point in partition, Z
    -- K: number of partitions in Z
    -- log_prior_fun: function of alpha. log P(alpha)
    """
    return gammaln(alpha) + float(K) * log(alpha) - gammaln(alpha + float(N))\
        + log_prior_fun(alpha)

def log_pflip(logp):
    """Categorical draw from a vector logp of log probabilities."""
    if len(logp) == 1:
        return 0
    P = log_normalize(logp)
    NP = np.exp(np.copy(P))
    assert math.fabs(1.0-sum(NP)) < 10.0**(-10.0)
    return pflip(NP)

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

def log_nchoosek(n, k):
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

def crp_gen(N, alpha):
    """Generates a random, N-length partition from the CRP with parameter
    alpha.
    """
    assert N > 0
    assert alpha > 0.0
    alpha = float(alpha)

    partition = np.zeros(N, dtype=int)
    Nk = [1]
    for i in range(1,N):
        K = len(Nk)
        ps = np.zeros(K+1)
        for k in range(K):
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

def bincount(X, bins=None):
    """Returns the frequency of each entry in bins of X. If bins is not
    specified, then bins is [0,1,..,max(X)].
    """
    Y = np.array(X, dtype=int)
    if bins == None:
        minval = np.min(Y)
        maxval = np.max(Y)
        bins = range(minval, maxval+1)

    counts = [0]*len(bins)

    for y in Y:
        bin_index = bins.index(y)
        counts[bin_index] += 1

    assert len(counts) == len(bins)
    assert sum(counts) == len(Y)

    return counts

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
        'normal_uc'         : False,
        'bernoulli'         : False,  # 0. or 1.
        'categorical'       : True,
        'lognormal'         : False,
        'poisson'           : False,
        'exponential'       : False,
        'exponential_uc'    : False,
        'geometric'         : True,
        'vonmises'          : False,
    }

    for i in xrange(len(X)):
        if is_discrete[cctypes[i]]:
            X[i] = np.array(X[i], dtype=int)
    return X
