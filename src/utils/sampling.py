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

from __future__ import division
from builtins import range
from past.utils import old_div
from math import fabs
from math import log

import numpy as np

from cgpm.utils import general as gu


def mh_sample(
        x, logpdf_target, jump_std, D,
        num_samples=1, burn=1, lag=1, rng=None):
    """Uses MH to sample from logpdf_target.

    Parameters
    ----------
    x : float
        Seed point.
    logpdf_target : function(x)
        Evaluates the log pdf of the target distribution at x.
    jump_std : float
        Standard deviation of jump distance, auto-tunes.
    D : tuple<float, float>
        Support of the target distribution.
    num_samples : int, optional
        Number of samples to return, default 1.
    burn : int, optional
        Number of samples to discard before any are collected, default 1.
    lag : int, optional
        Number of moves between successive samples, default 1.

    Returns
    -------
    samples : int or list
        If num_samples == 1 returns a float. Othewrise returns a
        `num_samples` length list.

    Example
    -------
    >>> # Sample from posterior of CRP(x) with exponential(1) prior
    >>> x = 1.0
    >>> logpdf_target = lambda x : gu.logp_crp(10, [5,3,2] , x) - x
    >>> jump_std = 0.5
    >>> D = (0.0, float('Inf'))
    >>> sample = mh_sample(x logpdf_target, jump_std, D)
    """
    assert D[0] <= x <= D[1]

    if rng is None:
        rng = gu.gen_rng()

    num_collected = 0
    iters = 0
    samples = []

    t_samples = num_samples * lag + burn

    checkevery = max(20, int(t_samples/100.0))
    accepted = 0.0
    acceptance_rate = 0.0
    iters = 1.0
    aiters = 1.0

    # XXX DISABLED.
    log_correction = lambda x, x_prime, jstd: 0

    # Is proposal symmetric? The folded normal is, but taking % 1?
    if D[0] == 0 and D[1] == 1:
        def jumpfun(x, jstd):
            x_prime = fabs(rng.normal(x, jstd))
            if x_prime > 1.0:
                x_prime = x_prime % 1
            return x_prime
    elif 0 <= D[0] and D[1] == float('inf'):
        jumpfun = lambda x, jstd: fabs(x + rng.normal(0.0, jstd))
    else:
        def jumpfun(x, jstd):
            MAX_TRIALS = 1000
            for _ in range(MAX_TRIALS):
                x_prime = rng.normal(x, jstd)
                if D[0] < x_prime < D[1]:
                    return x_prime
            raise RuntimeError('MH failed to rejection sample the proposal.')
        # XXX DISABLED, actually the mh function needs rewriting issue #56.
        def log_correction(x, x_prime, jstd):
            from scipy.stats import norm
            if D[0] == float('inf') and D[1] == float('inf'):
                return 0
            return norm.logcdf(old_div((D[1]-x),jump_std)-old_div((D[0]-x),jump_std)) \
                - norm.logcdf(old_div((D[1]-x_prime),jump_std)-old_div((D[0]-x_prime),jump_std))

    logp = logpdf_target(x)
    while num_collected < num_samples:

        x_prime = jumpfun(x, jump_std)
        assert D[0] < x_prime < D[1]
        logp_prime = logpdf_target(x_prime)

        # XXX DISABLED Correct MH sampler requires the log correction!
        # logp_corr = log_correction(x, x_prime, jump_std)

        if log(rng.rand()) < logp_prime - logp:
            x = x_prime
            logp = logp_prime
            accepted += 1.0
            acceptance_rate = old_div(accepted,aiters)

        if iters > burn and iters % lag == 0:
            num_collected += 1
            samples.append(x)

        # Keep the acceptance rate around .3 +/- .1.
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

def slice_sample(
        proposal_fun, log_pdf_fun, D,
        num_samples=1, burn=1, lag=1, w=1.0, rng=None):
    """Slice samples from the disitrbution defined by log_pdf_fun.

    Parameters
    ----------
    proposal_fun : function()
        Draws the initial point from proposal distribution.
    log_pdf_fun : function(x)
        Evaluates the log pdf of the target distribution at x.
    D : tuple<float, float>
        Support of the target distribution.
    num_samples : int, optional
        Number of samples to return, default 1.
    burn : int, optional
        Number of samples to discard before any are collected, default 1.
    lag : int, optional
        Number of moves between successive samples, default 1.

    Returns
    -------
    samples : int or list
        If num_samples == 1 returns a float. Othewrise returns a
        `num_samples` length list.

    Example:
    >>> # Sample from posterior of CRP(x) with exponential(x) prior
    >>> log_pdf_fun = lambda x : gu.lcrp(10, [5,3,2] , x) - x
    >>> proposal_fun = lambda : random.gammavariate(1.0,1.0)
    >>> D = (0.0, float('Inf'))
    >>> sample = slice_sample(proposal_fun, log_pdf_fun, D)
    """
    if rng is None:
        rng = gu.gen_rng()
    samples = []
    x = proposal_fun()
    f = lambda xp : log_pdf_fun(xp) # f is a log pdf
    num_iters = 0
    while len(samples) < num_samples:
        num_iters += 1
        u = log(rng.rand()) + f(x)
        r = rng.rand()
        a, b = _find_slice_interval(f, r, x, u, D, w=w)

        while True:
            x_prime = rng.uniform(a, b)
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

def _find_slice_interval(f, r, x, u, D, w=1.):
    """Given a point u between 0 and f(x), returns an approximated interval
    under f(x) at height u.
    """
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

def rejection_sample(
        target_pdf_fn, proposal_pdf_fn, proposal_draw_fn,
        N=1, rng=None):
    """Samples from target pdf using rejection sampling.

    Parameters
    ----------
    target_pdf_fn : function(x)
        Evaluates the log pdf of the target distribution at x.
    proposal_pdf_fn : function(x)
        Evaluates the log pdf of the proposal distribution at x. Should
        contain the target density, such that
        target_pdf_fn(x) <= proposal_fun(x) for all x.
    N : int, optional
        Number of samples to return, default 1.

    Returns
    -------
    samples : int or list
        If N == 1 returns a float. Othewrise returns an `N` length list.
    """
    if rng is None:
        rng = gu.gen_rng()

    samples = []

    while len(samples) < N:
        # Draw point along x-axis from proposal distribution.
        x = proposal_draw_fn()
        # Calculate proposal pdf at x.
        qx = proposal_pdf_fn(x)
        # Calculate pdf at x.
        px = target_pdf_fn(x)
        # Draw point randomly between 0 and qx.
        u = rng.rand()*qx
        # The proposal should contain the target for all x.
        assert px <= qx
        # If u is less than the target distribution pdf at x, then accept x
        if u < px:
            samples.append(x)

    if N == 1:
        return samples[0]
    else:
        return samples

def resample_data(state):
    """Samples and resets data in the state. XXX Currently broken."""
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
