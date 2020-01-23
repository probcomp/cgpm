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

'''
This module contains implementations of simulate and logpdf specialized to
cgpm.crosscat, avoiding overhead of recursive implementations using the
importance network on the sub-cgpms that comprise cgpm.crosscat.State.
'''

from builtins import zip
from itertools import chain

import numpy as np

from cgpm.primitives.crp import Crp

from cgpm.utils.general import log_normalize
from cgpm.utils.general import log_pflip
from cgpm.utils.general import logsumexp
from cgpm.utils.general import merged

from cgpm.utils.validation import partition_query_evidence


def state_logpdf(state, rowid, targets, constraints=None):
    targets_lookup, constraints_lookup = partition_query_evidence(
        state.Zv(), targets, constraints)
    logps = (
        view_logpdf(
            view=state.views[v],
            rowid=rowid,
            targets=targets_lookup[v],
            constraints=constraints_lookup.get(v, {})
        )
        for v in targets_lookup
    )
    return sum(logps)


def state_simulate(state, rowid, targets, constraints=None, N=None):
    targets_lookup, constraints_lookup = partition_query_evidence(
        state.Zv(), targets, constraints)
    N_sim = N if N is not None else 1
    draws = (
        view_simulate(
            view=state.views[v],
            rowid=rowid,
            targets=targets_lookup[v],
            constraints=constraints_lookup.get(v, {}),
            N=N_sim
        )
        for v in targets_lookup
    )
    samples = [merged(*l) for l in zip(*draws)]
    return samples if N is not None else samples[0]


def view_logpdf(view, rowid, targets, constraints):
    if not view.hypothetical(rowid):
        return _logpdf_row(view, targets, view.Zr(rowid))
    Nk = view.Nk()
    N_rows = len(view.Zr())
    K = view.crp.clusters[0].gibbs_tables(-1)
    lp_crp = [Crp.calc_predictive_logp(k, N_rows, Nk, view.alpha()) for k in K]
    lp_constraints = [_logpdf_row(view, constraints, k) for k in K]
    if all(np.isinf(lp_constraints)):
        raise ValueError('Zero density constraints: %s' % (constraints,))
    lp_cluster = log_normalize(np.add(lp_crp, lp_constraints))
    lp_targets = [_logpdf_row(view, targets, k) for k in K]
    return logsumexp(np.add(lp_cluster, lp_targets))


def view_simulate(view, rowid, targets, constraints, N):
    if not view.hypothetical(rowid):
        return _simulate_row(view, targets, view.Zr(rowid), N)
    Nk = view.Nk()
    N_rows = len(view.Zr())
    K = view.crp.clusters[0].gibbs_tables(-1)
    lp_crp = [Crp.calc_predictive_logp(k, N_rows, Nk, view.alpha()) for k in K]
    lp_constraints = [_logpdf_row(view, constraints, k) for k in K]
    if all(np.isinf(lp_constraints)):
        raise ValueError('Zero density constraints: %s' % (constraints,))
    lp_cluster = np.add(lp_crp, lp_constraints)
    ks = log_pflip(lp_cluster, array=K, size=N, rng=view.rng)
    counts = {k:n for k,n in enumerate(np.bincount(ks)) if n > 0}
    samples = (_simulate_row(view, targets, k, counts[k]) for k in counts)
    return chain.from_iterable(samples)


def _logpdf_row(view, targets, cluster):
    """Return joint density of the targets in a fixed cluster."""
    return sum(
        view.dims[c].logpdf(None, {c:x}, None, {view.outputs[0]: cluster})
        for c, x in targets.items()
    )


def _simulate_row(view, targets, cluster, N):
    """Return sample of the targets in a fixed cluster."""
    samples = (
        view.dims[c].simulate(None, [c], None, {view.outputs[0]: cluster}, N)
        for c in targets
    )
    return (merged(*l) for l in zip(*samples))
