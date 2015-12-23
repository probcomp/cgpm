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

import copy
from math import fabs, log

import numpy as np
from numpy.random import normal
from scipy.misc import logsumexp

import gpmcc.utils.general as gu

def mh_sample(x, log_pdf_lambda, jump_std, D, num_samples=1, burn=1, lag=1):
    """Uses MH to sample from log_pdf_lambda.

    Arguments:
    ... x: seed point
    ... log_pdf_lambda: function that evaluates the log pdf at x
    ... jump_std: standard deviation of jump distance (tunes itself)
    ... D: domain

    Keyword arguments:
    ... num_samples: number of samples to take
    ... burn: samples to throw out before any are collected
    ... lag: moves between samples

    Returns:
    ... If num_samples == 1, returns a float, else resturns an num_samples length
        list

    Example:
    >>> # Sample from posterior of CRP(x) with exponential(1) prior
    >>> x = 1.0
    >>> log_pdf_lambda = lambda x : gu.lcrp(10, [5,3,2] , x) - x
    >>> jump_std = 0.5
    >>> D = (0.0,float('Inf'))
    >>> sample = mh_sample(x log_pdf_lambda, jump_std, D)
    """
    num_collected = 0
    iters = 0
    samples = []

    t_samples = num_samples * lag + burn

    checkevery = max(20, int(t_samples/100.0))
    accepted = 0.0
    acceptance_rate = 0.0
    iters = 1.0
    aiters = 1.0

    if D[0] >= 0.0 and D[1] == float('Inf'):
        jumpfun = lambda x, jstd: fabs(x + normal(0.0, jstd))
    elif D[0] == 0 and D[1] == 1:
        def jumpfun(x, jstd):
            x = fabs(x + normal(0.0, jstd))
            if x > 1.0:
                x = x%1
            assert x > 0 and x < 1
            return x
    else:
        jumpfun = lambda x, jstd: x + normal(0.0, jstd)

    logp = log_pdf_lambda(x)
    while num_collected < num_samples:

        # every now and then propose wild jumps incase there very distant modes
        x_prime = jumpfun(x, jump_std)
        assert  x_prime > D[0] and x_prime < D[1]
        logp_prime = log_pdf_lambda(x_prime)

        # if log(random.random()) < logp_prime - logp:
        if log(np.random.random()) < logp_prime - logp:
            x = x_prime
            logp = logp_prime
            accepted += 1.0
            acceptance_rate = accepted/aiters

        if iters > burn and iters%lag == 0:
            num_collected += 1
            samples.append(x)

        # keep the acceptance rate around .3 +/- .1
        if iters % checkevery == 0:
            if acceptance_rate >= .4:
                jump_std *= 1.1
            elif acceptance_rate <= .2:
                jump_std *= .9019
            # print("j : %1.4f, AR: %1.4f" % (jump_std, acceptance_rate))
            accepted = 0.0
            acceptance_rate = 0.0
            aiters = 0.0

        iters += 1.0
        aiters += 1.0

    if num_samples == 1:
        return samples[0]
    else:
        return samples

def slice_sample(proposal_fun, log_pdf_lambda, D, num_samples=1, burn=1, lag=1,
        w=1.0):
    """Slice samples from the disitrbution defined by log_pdf_lambda.

    Arguments:
    ... proposal_fun: function that draws the initial point e.g. random.random
    ... log_pdf_lambda: function that evaluates the log pdf at x
    ... D: tuple. The domain.

    Keyword arguments:
    ... num_samples: number of samples to take
    ... burn: samples to throw out before any are collected
    ... lag: moves between samples

    Returns:
    ... If num_samples == 1, returns a float, else resturns an num_samples length list

    Example:
    >>> # Sample from posterior of CRP(x) with exponential(x) prior
    >>> log_pdf_lambda = lambda x : gu.lcrp(10, [5,3,2] , x) - x
    >>> proposal_fun = lambda : random.gammavariate(1.0,1.0)
    >>> D = (0.0, float('Inf'))
    >>> sample = slice_sample(proposal_fun, log_pdf_lambda, D)
    """
    samples = []
    x = proposal_fun()
    f = lambda xp : log_pdf_lambda(xp) # f is a log pdf
    num_iters = 0
    while len(samples) < num_samples:
        num_iters += 1
        u = log(np.random.random()) + f(x)
        a, b = _find_slice_interval(f, x, u, D, w=w)

        while True:
            x_prime = np.random.uniform(a, b)
            if f(x_prime) > u:
                x = x_prime
                break
            else:
                if x_prime > x:
                    b = x_prime
                else:
                    a = x_prime;

        if num_iters >= burn and num_iters%lag == 0:
            samples.append(x)

    if num_samples == 1:
        return samples[0]
    else:
        return samples

def _find_slice_interval(f, x, u, D, w=1.0):
    """Given a point u between 0 and f(x), returns an approximated interval
    under f(x) at height u.
    """
    r = np.random.random()
    a = x - r*w
    b = x + (1-r)*w

    if a < D[0]:
        a = D[0]
    else:
        while f(a) > u:
            a -= w
            if a < D[0]:
                a = D[0]
                break

    if b > D[1]:
        b = D[1]
    else:
        while f(b) > u:
            b += w
            if b > D[1]:
                b = D[1]
                break
    return a, b

def rejection_sample(target_pdf_fn, proposal_pdf_fn, proposal_draw_fn, N=1):
    """Samples from target pdf using rejection sampling.

    Input arguments:
    -- target_pdf_fn: the target distribution pdf. Should take a single
    argument, x.
    -- proposal_pdf_fn: the propsal distribution pdf. Should take a single
    argument, x. Should also contain the target for all x, that is,
    target_pdf_fn(x) <= proposal_pdf_fn(x)
    -- proposal_draw_fn: draws x randomly from the domain of the target pdf
    according to the proposal distribution.

    Keyword Arguments:
    -- N: the number of samples to take.

    NOTES:
    This was specifically implemented for the VonMiese predictive distribution
    which, because it is cyclic, is not super obvious to me how to apply
    adaptive rejection sampling strategies. This is wasteful, but it's correct
    and quicker than esitimating the CDF for inversion sampling.
    """
    samples = []

    while len(samples) < N:
        # Draw point along X-axis from proposal distribution.
        x = proposal_draw_fn()
        # Calculate proposal pdf at x.
        qx = proposal_pdf_fn(x)
        # Calculate pdf at x.
        px = target_pdf_fn(x)
        # Draw point randomly between 0 and qx.
        u = np.random.random()*qx
        # The proposal should contain the target for all x.
        assert px <= qx
        # If u is less than the target distribution pdf at x, then accept x
        if u < px:
            samples.append(x)

    if N == 1:
        return samples[0]
    else:
        return samples

def simple_predictive_probability(state, row, col, X):
    logps = np.zeros(len(X))
    i = 0
    for x in X:
        is_observed = row > state.n_rows
        if is_observed:
            logp = _simple_predictive_probability_unobserved(state, col, x)
        else:
            logp = _simple_predictive_probability_observed(state, row, col, x)

        logps[i] = logp
        i += 1
    if i == 1:
        return logps[0]
    else:
        return logps

def _simple_predictive_probability_observed(state, row, cols, x):
    cluster = create_cluster(state, row, cols)
    logp = cluster.predictive_logp(x)
    return logp

def _simple_predictive_probability_unobserved(state, col, x):
    log_pK = compute_cluster_crp_logps(state, col)
    clusters = create_clusters(state, col)
    logps = []
    for cluster in clusters:
        logps.append(cluster.predictive_logp(x))
    logps = np.array(logps) + log_pK
    return logsumexp(logps)

def simulate(state, rowid, query, evidence=None, N=1):
    # TODO: accept multipe queries and evidence. Return float if N=1
    if 0 <= rowid < state.n_rows and float(rowid) == int(rowid):
        samples = simulate_observed(state, rowid, query, N=N)
    else:
        samples = simulate_unobserved(state, query, evidence, N=N)
    return samples

def simulate_observed(state, rowid, query, N=1):
    """Simulates a hypothetical member, with the same latents as rowid."""
    samples = []
    for _ in xrange(N):
        row_draw = []
        for col in query:
            cluster = create_cluster(state, rowid, col)
            row_draw.append(cluster.simulate())
        samples.append(row_draw)
    return samples

def simulate_unobserved(state, query, evidence=None, N=1):
    """Simulates a hypothetical member, with no observed latents."""
    # Default parameter.
    if evidence is None:
        evidence = []

    # Obtain all views of the query columns.
    views = [state.Zv[col] for col in query]

    # Create component models for all the queries.
    clusters_for = dict()
    for col in query:
        clusters_for[col] = create_clusters(state, col)

    cols_in = dict()
    for v in set(views):
        cols_in[v] = [col for col in query if state.Zv[col] == v]

    # Obtain the probability of hypothetical row belonging to each cluster.
    clusterps_for = dict()
    for v in set(views):
        # CRP densities.
        logp_crp = compute_cluster_crp_logps(state, v)
        # Posterior predictive density of evidence.
        logp_data = np.zeros(len(logp_crp))
        for (col, val) in evidence:
            if state.Zv[col] == v:
                logp_data += compute_cluster_data_logps(state, col, val)
        clusterps_for[v] = np.exp(gu.log_normalize(logp_crp + logp_data))

    samples = []
    for _ in xrange(N):
        draw = []
        for v in set(views):
            # Sample a cluster.
            k = gu.pflip(clusterps_for[v])
            for col in cols_in[v]:
                # Sample data from the cluster.
                x = clusters_for[col][k].simulate()
                draw.append(x)
        samples.append(draw)

    return np.asarray(samples)

def compute_cluster_crp_logps(state, view):
    log_crp_numer = state.views[view].Nk[:]
    log_crp_numer.append(state.views[view].alpha)   # singleton cluster
    log_crp_denom = log(state.n_rows + state.views[view].alpha)
    cluster_crps = np.log(np.array(log_crp_numer))-log_crp_denom
    return cluster_crps

def compute_cluster_data_logps(state, col, x):
    """Computes the Pr[x|z=k] for each cluster k in col, including singleton."""
    clusters = create_clusters(state, col)
    logps = np.zeros(len(clusters))
    for i, cluster in enumerate(clusters):
        logps[i] = cluster.predictive_logp(x)
    return logps

def create_clusters(state, col):
    """Returns a list of all the clusters in the dim, plus one singleton."""
    hypers = state.dims[col].hypers
    # Existing cluster.
    clusters = copy.deepcopy(state.dims[col].clusters)
    # Singleton cluster.
    singleton_cluster = state.dims[col].model(distargs=state.dims[col].distargs,
        **hypers)
    clusters.append(singleton_cluster)
    return clusters

def create_cluster(state, rowid, col):
    """Returns the exact cluster of the cell (rowid, col)."""
    v = state.Zv[col]
    k = state.views[v].Zr[rowid]
    cluster = copy.deepcopy(state.dims[col].clusters[k])
    return cluster

def resample_data(state):
    """Samples and resets data in the state."""
    n_rows = state.n_rows
    n_cols = state.n_cols
    table = np.zeros( (n_rows, n_cols) )
    # state.clear_data()

    all_rows = [r for r in range(n_rows)]
    np.random.shuffle(all_rows)
    for col in range(n_cols):
        for row in all_rows:
            # get the view and cluster to which the datum is assigned
            view = state.Zv[col]
            cluster = state.views[view].Z[row]
            # sample a new element
            x = simple_predictive_sample(state, int(row), col)[0]
            # remove the current element
            state.dims[col].remove_element(row, cluster)
            # replace the current table element with the new element
            state.dims[col].X[row] = x
            # insert the element into the cluster
            state.dims[col].insert_element(row, cluster)
            # store
            table[row,col] = x

    X = []
    for col in range(n_cols):
        N = 0
        for cluster in state.dims[col].clusters:
            N += cluster.N
        assert N == n_rows
        X.append(table[:,col].flatten(1))

    return X
