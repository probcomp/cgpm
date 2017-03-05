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


import numpy as np

from cgpm.utils.general import logsumexp
from cgpm.utils.config import check_env_debug


def relevance_probability(view, rowid_target, rowid_query):
    """Compute probability of customers in same table.

    Given a single target rowid T and list of query rowids Q, compute the
    posterior probability that T and all rowids in Q are assigned to the same
    table, conditioned on all rowids in Q being assigned to the same table as
    well as the row data values xT and xQ.

    Let S be the event of all rowids in Q are assigned to the same table:
        S = [zQ[0] = zQ[1] = ... = zQ[-1]]

    The first quantity of interest is;

        Pr[zT = zQ | xT, xQ, S] = Pr[zT = zQ, xT, xQ, S] / Pr[xT, xQ, S]

        The numerator is:

            Pr[zT = zQ, xT, xQ, S]
              = \sum_k Pr[zT=k, zQ=k, xT, xQ]
              = \sum_k Pr[xT, xQ | zT=K, zQ=k] * Pr[zT=k, zQ=k]

        where k is list of tables in the CRP plus a fresh singleton.

    The second quantity of interest is:
        Pr[zT \ne zQ | xT, xQ, S] = Pr[zT \ne zQ, xT, xQ, S] / Pr[xT, xQ, S]

        The numerator is:

            Pr[zT \ne zQ, xT, xQ, S]
              = \sum_kT \sum_kQ|kT Pr[zT=kT, zQ=kQ, xT, xQ]
              = \sum_kT \sum_kQ|kT Pr[xT, xQ | zT=kT, zQ=kQ] * Pr[zT=kT, zQ=kQ]

        where kT is list of tables in the CRP plus a fresh singleton, and
        kQ|kT in the inner sum is all tables in the CRP other than kT (plus a
        fresh singleton when kT is itself a singleton).

        For example if the tables are [0, 1] then:
            kT = [0, 1, 2]
            kQ|kT = [[1, 2], [0, 2], [0,1,3]

    If computation is correct then the first and second quantities are equal
    to the normalizer, which is given by:

            Pr[xT, xQ, S]
              = \sum_kQ Pr[zQ[0]=kQ, ..., zQ[-1]=kQ, xT, xQ]
              = \sum_kQ Pr[xT, xQ|zQ] * Pr[zQ[0]=kQ, ..., zQ[-1]=kQ]
              = \sum_kQ (\sum_kT Pr[xT, zT=kT])
                  Pr[xQ|zQ] * Pr[zQ[0]=kQ, ..., zQ[-1]=kQ]
              = \sum_kQ (\sum_kT Pr[xT|zT=kT] * Pr[zT=kT| zQ=kQ])
                  Pr[xQ|zQ] * Pr[zQ[0]=kQ, ..., zQ[-1]=kQ]

        where kQ is list of tables in the CRP plus a fresh singleton.

        The inner sum over kT computes the predictive density of xT when all the
        rows in Q are in table kQ marginalizing over all assignments.

    Parameters
    ----------
    view : cgpm.mixtures.View
        View CGPM representing the DP mixture.
    rowid_target : int
        The target rowid, must be incorporate in the view.
    rowid_query : list<int>
        The query rowids, must be incorporated in the view.

    Returns
    -------
    relevance_probability : float
        The posterior probability the target is in the same cluster as query.
    """

    if len(rowid_query) < 1:
        raise ValueError('No query rows:, %s' % (rowid_query))
    if rowid_target in rowid_query:
        return 1.

    # Retrieve target crp assignments and data to restore later.
    assignments_target = view.Zr(rowid_target)
    values_target = row_values(view, rowid_target)

    # Retrieve query crp assignments and data to restore later.
    values_query = [row_values(view, r) for r in rowid_query]
    assignments_query = [view.Zr(r) for r in rowid_query]

    # Retrieve view logpdf to verify no mutation afterwards.
    if check_env_debug():
        logpdf_score_full = view.logpdf_score()

    # Unincorporate target and query rows.
    view.unincorporate(rowid_target)
    for rowid_q in rowid_query:
        view.unincorporate(rowid_q)

    # Retrieve current tables.
    tables_crp = sorted(view.crp.clusters[0].counts)

    # Retrieve cluster-wise marginal likelhoods.
    tables_same = get_tables_same(tables_crp)
    logps_clusters = [get_view_logpdf_score(view, t, t) for t in tables_same]

    # Compute Pr[xT, xQ, S]
    #   = \sum_kT \sum_kQ Pr[zT=kT, zQ=kQ, xT, xQ]
    #   = \sum_kT \sum_kQ Pr[xT, xQ | zT=kT, zQ=kQ] * Pr[zT=kT, zQ=kQ]
    logps_condition = [
        logpdf_assignments_marginalize_target(
            view,
            rowid_query,
            rowid_query,
            values_target,
            values_query,
            table_query
        )
        for table_query in tables_same
    ]
    logp_condition = logsumexp(np.subtract(logps_condition, logps_clusters))

    # Compute Pr[zT = zQ, xT, xQ, S]
    #   = \sum_k Pr[zT=k, zQ=k, xT, xQ]
    #   = \sum_k Pr[xT, xQ | zT=K, zQ=k] * Pr[zT=k, zQ=k]
    logps_same_table = [
        logpdf_assignments(
            view,
            rowid_target,
            rowid_query,
            values_target,
            values_query,
            table,
            table,
        )
        for table in tables_same
    ]
    logp_same_table = logsumexp(np.subtract(logps_same_table, logps_clusters))

    # ----------------------------------------------------------------------
    # The following computation is not necessary and introduces O(K^2)
    # overhead due to the nested sum, but serves as a vital check for
    # correct implementation (as noted in the docstring.)
    # ----------------------------------------------------------------------
    # Compute Pr[zT \ne zQ, xT, xQ, S]
    #   = \sum_kT \sum_kQ|kT Pr[zT=kT, zQ=kQ, xT, xQ]
    #   = \sum_kT \sum_kQ|kT Pr[xT, xQ | zT=kT, zQ=kQ] * Pr[zT=kT, zQ=kQ]
    if check_env_debug():
        tables_target, tables_query = get_tables_different(tables_crp)
        # Compute the base logps.
        logps_clusters_diff = [
            [
                get_view_logpdf_score(
                    view,
                    table_target,
                    table_q,
                )
                for table_q in table_query
            ]
            for table_target, table_query in zip(tables_target, tables_query)
        ]
        # Compute the new logps.
        logps_diff_table = [
            [
                logpdf_assignments(
                    view,
                    rowid_target,
                    rowid_query,
                    values_target,
                    values_query,
                    table_target,
                    table_q,
                )
                for table_q in table_query
            ]
            for table_target, table_query in zip(tables_target, tables_query)
        ]
        # Compute the deltas.
        logps_delta = [
            np.subtract(a, b)
            for (a, b) in zip(logps_diff_table, logps_clusters_diff)
        ]
        # Sum the deltas.
        logp_diff_table = logsumexp([logsumexp(l) for l in logps_delta])

        # Confirm logp_same_table + logp_diff_table equal normalizing constant.
        assert np.allclose(
            logsumexp([logp_same_table, logp_diff_table]),
            logp_condition
        )

    # Restore the target row.
    values_target[view.outputs[0]] = assignments_target
    view.incorporate(rowid_target, values_target)

    # Restore the query rows.
    for rowid, values, z in zip(rowid_query, values_query, assignments_query):
        values[view.outputs[0]] = z
        view.incorporate(rowid, values)

    # Confirm no mutation has occured.
    if check_env_debug():
        assert np.allclose(view.logpdf_score(), logpdf_score_full)

    # Return the relevance probability.
    return np.exp(logp_same_table - logp_condition)


def logpdf_assignments_marginalize_target(
        view, rowid_target, rowid_query, values_target, values_query,
        table_query):
    """Compute the joint probability of crp assignment and data for the
    query rows, marginalizing over the assignment of the target row."""

    # Incorporate the query rows.
    for rowid_q, values_q in zip(rowid_query, values_query):
        values_q[view.outputs[0]] = table_query
        view.incorporate(rowid_q, values_q)
        del values_q[view.outputs[0]]

    # Compute new marginal likelihood.
    logpdf_clusters = get_view_logpdf_score(view, table_query, table_query)

    # Compute predictive probability of target (marginalizes over tables).
    logpdf_predictive = view.logpdf(-1, values_target)

    # Unincorporate the query.
    for rowid_q in rowid_query:
        view.unincorporate(rowid_q)

    # Predictive joint probability (see docstring).
    return logpdf_predictive + logpdf_clusters


def logpdf_assignments(
        view, rowid_target, rowid_query, values_target, values_query,
        table_target, table_query):
    """Compute the joint probability of crp assignment and data."""

    # Incorporate target row.
    values_target[view.outputs[0]] = table_target
    view.incorporate(rowid_target, values_target)
    del values_target[view.outputs[0]]

    # Incorporate query rows.
    for rowid_q, row_values_q in zip(rowid_query, values_query):
        row_values_q[view.outputs[0]] = table_query
        view.incorporate(rowid_q, row_values_q)
        del row_values_q[view.outputs[0]]

    # Compute new marginal likelihood.
    logpdf_clusters = get_view_logpdf_score(view, table_target, table_query)

    # Unincorporate the target and query rows.
    view.unincorporate(rowid_target)
    for rowid_q in rowid_query:
        view.unincorporate(rowid_q)

    # Predictive joint probability is difference in marginal likelihood.
    return logpdf_clusters


def row_values(view, rowid):
    """Retrieve observations for rowid, only considering variables in view."""
    return {output: view.X[output][rowid] for output in view.outputs[1:]}


def get_tables_same(tables):
    """Return tables to iterate over when query, target in same table."""
    singleton = max(tables) + 1
    return tables + [singleton]


def get_tables_different(tables):
    """Return tables to iterate over when query, target in different table."""
    singleton = max(tables) + 1
    tables_target = tables + [singleton]
    auxiliary_table = lambda t: [] if t < singleton else [singleton+1]
    tables_query = [
        filter(lambda x: x != t, tables_target) + auxiliary_table(t)
        for t in tables_target
    ]
    return tables_target, tables_query


def get_view_logpdf_score(view, table_target, table_query):
    """Retrieve logpdf score restricted to the crp and specified clusters."""
    logpdf_crp = view.crp.logpdf_score()
    logpdf_clusters = get_cluster_logpdf_score(view, table_query)
    if table_target != table_query:
        logpdf_clusters += get_cluster_logpdf_score(view, table_target)
    return logpdf_crp + logpdf_clusters


def get_cluster_logpdf_score(view, k):
    """Return marginal likelihood of cluster k in View (0 for fresh cluster)."""
    return sum([
        d.clusters[k].logpdf_score() for d in view.dims.itervalues()
    ]) if k in view.crp.clusters[0].counts else 0
