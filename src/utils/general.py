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

import math
import warnings
import sys
from math import log

import numpy as np
from scipy.special import i0 as bessel_0
from scipy.misc import logsumexp
from scipy.special import gammaln

from gpmcc.utils import validation as vu

colors = ['red', 'blue', 'green', 'magenta', 'orange', 'purple', 'brown',
    'black']

def gen_rng(seed=None):
    if seed is None:
        seed = np.random.randint(sys.maxsize)
    return np.random.RandomState(seed)

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

def normalize(p):
    """Normalizes a np array of probabilites."""
    return np.asarray(p, dtype=float) / sum(p)

def logp_crp(N, Nk, alpha):
    """Returns the log normalized P(N,K|alpha), where N is the number of
    customers and K is the number of tables.
    https://www.cs.princeton.edu/~blei/papers/GershmanBlei2012.pdf#page=4 (eq 8)
    """
    return len(Nk)*log(alpha) + np.sum(gammaln(Nk)) \
        + gammaln(alpha) - gammaln(N+alpha)

def logp_crp_unorm(N, K, alpha):
    """Returns the log unnormalized P(N,K|alpha), where N is the number of
    customers and K is the number of tables. Use for effeciency to avoid
    computing terms that are not a function of alpha.
    """
    return K*log(alpha) + gammaln(alpha) - gammaln(N+alpha)

def logp_crp_gibbs(Nk, Z, i, alpha, m):
    """Compute the CRP probabilities for a Gibbs transition of customer i,
    with table counts Nk, table assignments Z, and m auxiliary tables."""
    singleton = Nk[Z[i]] == 1
    m_aux = m-1 if singleton else m

    p_table_aux = alpha/float(m)

    p_current = lambda : p_table_aux if singleton else Nk[Z[i]]-1
    p_other = lambda t : Nk[t]
    p_table = lambda t: p_current() if t == Z[i] else p_other(t)

    logp_tables = [log(p_table(t)) for t in xrange(len(Nk))]
    logp_aux = [log(p_table_aux) for _ in xrange(m_aux)]

    return logp_tables + logp_aux

def log_pflip(logp, rng=None):
    """Categorical draw from a vector logp of log probabilities."""
    if len(logp) == 1:
        return 0
    p = np.exp(log_normalize(logp))
    return pflip(p, rng=rng)

def pflip(p, rng=None):
    """Categorical draw from a vector p of probabilities."""
    if rng is None:
        rng = gen_rng()
    if len(p) == 1:
        return 0
    p = normalize(p)
    if 10.**(-8.) < math.fabs(1.-sum(p)):
        warnings.warn('pflip probability vector sums to %f.' % sum(p))
    return rng.choice(range(len(p)), size=1, p=p)[0]

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
    assert N > 0 and alpha > 0.
    alpha = float(alpha)

    partition = np.zeros(N, dtype=int)
    Nk = [1]
    for i in xrange(1,N):
        K = len(Nk)
        ps = np.zeros(K+1)
        for k in xrange(K):
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

def simulate_crp_constrained(N, alpha, Cd, Ci, Rd, Ri):
    """Simulates a CRP with N customers and concentration alpha. Cd is a list,
    where each entry is a list of friends. Ci is a list of tuples, where each
    tuple is a pair of enemies."""
    vu.validate_crp_constrained_input(N, Cd, Ci, Rd, Ri)
    assert N > 0 and alpha > 0.

    # Initial partition.
    Z = -1 * np.ones(N, dtype=int)

    # Friends dictionary from Cd.
    friends = {col:block for block in Cd for col in block}

    # Assign customers.
    for cust in xrange(N):
        if Z[cust] > -1: continue
        # Find valid tables for cust and friends.
        assert all(Z[f] == -1 for f in friends.get(cust, [cust]))
        prob_table = [0] * (max(Z)+1)
        for t in xrange(max(Z)+1):
            # Current customers at table t.
            t_custs = [i for i,z in enumerate(Z) if z==t]
            prob_table[t] = len(t_custs)
            # Does f \in {cust \union cust_friends} have an enemy in table t?
            for tc in t_custs:
                for f in friends.get(cust, [cust]):
                    if not vu.check_compatible_customers(Cd,Ci,Ri,Rd,f,tc):
                        prob_table[t] = 0
                        break
        # Choose from valid tables using CRP.
        prob_table.append(alpha)
        assignment = pflip(prob_table)
        for f in friends.get(cust, [cust]):
            Z[f] = assignment

    # At most N tables.
    assert all(0 <= t < N for t in Z)
    assert vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)
    return Z

def build_rowid_blocks(Zvr):
    A = np.asarray(Zvr).T
    U = map(tuple, A)
    return {u:np.where(np.all(A==u, axis=1))[0] for u in U}
