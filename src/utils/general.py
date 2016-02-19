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
import warnings
from math import log

import networkx as nx
import numpy as np
from scipy.special import i0 as bessel_0
from scipy.misc import logsumexp
from scipy.special import gammaln

from gpmcc.utils import validation as vu

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
    if not math.fabs(1.0-sum(p)) < 10.0**(-8.0):
        warnings.warn('log_pflip probability vector sums to {}.'.format(sum(p)))
    return pflip(p)

def pflip(p):
    """Categorical draw from a vector p of probabilities."""
    if len(p) == 1:
        return 0
    p = np.asarray(p).astype(float)
    p /= sum(p)
    if not math.fabs(1.0-sum(p)) < 10.0**(-8.0):
        warnings.warn('pflip probability vector sums to {}.'.format(sum(p)))
    return np.random.choice(range(len(p)), size=1, p=p)[0]

def log_linspace(a, b, n):
    """linspace from a to b with n entries over log scale (mor entries at
    smaller values).
    """
    return np.exp(np.linspace(log(a), log(b), n))

def log_nCk(n, k):
    """log(choose(n,k)) with overflow protection."""
    if n == 0 or k == 0 or n == k:
        return 0
    return log(n) + gammaln(n) - log(k) - gammaln(k) - log(n-k) - gammaln(n-k)

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
    return np.array(partition)

def simulate_crp_constrained(N, alpha, Cd, Ci):
    """Simulates a CRP with N customers and concentration alpha. Cd is a list,
    where each entry is a list of friends. Ci is a list of tuples, where each
    tuple is a pair of enemies."""
    vu.validate_crp_constrained_input(N, Cd, Ci)
    # Initial partition.
    Zv = -1 * np.ones(N, dtype=int)

    # Neighbors dictionary from Cd.
    neighbors = {col:block for block in Cd for col in block}

    # Minimum number of views is largest connected component in Ci.
    G = nx.Graph(data=Ci)
    components = list(sorted(nx.connected_components(G), key=len, reverse=True))

    # Independent columns in largest component all in seperate views.
    if components:
        for i, col in enumerate(components[0]):
            assert Zv[col] == -1
            # Create a view with col and all its friends.
            friends = neighbors.get(col, [col])
            for f in friends:
                assert Zv[f] == -1
                Zv[f] = i

    # Assign remaining columns.
    for col in xrange(N):
        if Zv[col] > -1:
            continue
        # Find valid views for this column and all its friends.
        friends = neighbors.get(col, [col])
        assert all(Zv[f] == -1 for f in friends)
        prob_view = [0] * (max(Zv)+1)
        for v in xrange(max(Zv)+1):
            # All columns in valid_views[v].
            v_cols = [i for i,z in enumerate(Zv) if z==v]
            assert len(v_cols) > 0
            prob_view[v] = len(v_cols)
            # Are there are contradictions between v_cols and friends?
            for v_col in v_cols:
                if any((f, v_col) in Ci or (v_col, f) in Ci for f in friends):
                    prob_view[v] = 0
                    break
        # Choose from valid_view using CRP.
        prob_view.append(alpha)
        assignment = pflip(prob_view)
        for f in friends:
            Zv[f] = assignment

    assert all(0 <= v < N for v in Zv)
    return Zv
