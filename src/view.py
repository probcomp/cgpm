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

from math import log
from scipy.misc import logsumexp

import numpy as np
import gpmcc.utils.general as gu
from gpmcc.dim import Dim

class View(object):
    """View, a collection of Dim and their row mixtures."""

    def __init__(self, X, dims, alpha=None, Zr=None, rng=None):
        """View constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data and optional row partition.

        Parameters
        ----------
        X : np.ndarray
            Global dataset of dimension N x D. The invariant is that
            the data for dim.index should be in X[:,dim.index] and the data
            for rowid should X[rowid,:]. All rows in X will be incorporated.
        dims : list<Dim>
            A list of Dim objects in this View.
        alpha : float, optional
            CRP concentration parameter. If None, selected from grid uniformly
            at random.
        Zr : list<int>, optional
            Starting partiton of rows to categories where Zr[i] is the latent
            clsuter of row i. If None, is sampled from CRP(alpha).
        """
        # Entropy.
        self.rng = gu.gen_rng() if rng is None else rng

        # Dataset.
        self.X = X

        # Generate alpha.
        self.alpha_grid = gu.log_linspace(1./len(self.X), len(self.X), 30)
        if alpha is None:
            alpha = self.rng.choice(self.alpha_grid)
        self.alpha = alpha

        # Generate row partition.
        if Zr is None:
            Zr = gu.simulate_crp(len(self.X), alpha, rng=self.rng)
        self.Zr = list(Zr)
        self.Nk = list(np.bincount(Zr))

        # Incoroprate the dimensions.
        self.dims = dict()
        for dim in sorted(dims, key=lambda d: d.is_conditional()):
            self.incorporate_dim(dim)

        # self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, dim, reassign=True):
        """Incorporate the dim into this View. If reassign is False, the row
        partition of dim should match self.Zr already."""
        self.dims[dim.index] = dim
        if reassign:
            Y = None
            if dim.is_conditional():
                Y = self._unconditional_values()
                dim.distargs['cctypes'] = self._unconditional_cctypes()
                dim.distargs['ccargs'] = self._unconditional_ccargs()
            dim.bulk_incorporate(self.X[:, dim.index], self.Zr, Y=Y)
        return sum(dim.logpdf_marginal())

    def unincorporate_dim(self, dim):
        """Remove dim from this View (does not modify dim)."""
        del self.dims[dim.index]
        return sum(dim.logpdf_marginal())

    def incorporate_row(self, rowid, k=None):
        """Incorporate rowid from the global dataset X into the view. Use
        set_dataset to update X if it has new rowids.

        Parameters
        ----------
        rowid : int
            The rowid in dataset X to be incorporated.
        k : int, optional
            Index of the cluster to assign the row. If unspecified, will be
            sampled. If 0 <= k < len(view.Nk) will insert into an existing
            cluster. If k = len(state.Nv) a singleton cluster will be created.
        """
        # If k unspecified, transition the new rowid.
        k = 0 if k is None else k
        transition = [rowid] if k is None else []
        # Incorporate into dims.
        for dim in self.dims.values():
            dim.incorporate(self.X[rowid, dim.index], k)
        # Account.
        if k == len(self.Nk):
            self.Nk.append(0)
        self.Nk[k] += 1
        if rowid == len(self.Zr):
            self.Zr.append(0)
        self.Zr[rowid] = k
        self.transition_rows(rows=transition)

    def unincorporate_row(self, rowid):
        """Remove rowid from the global datset X from this view."""
        # Unincorporate from dims.
        for dim in self.dims.values():
            dim.unincorporate(self.X[rowid, dim.index], self.Zr[rowid])
        # Account.
        k = self.Zr[rowid]
        self.Nk[k] -= 1
        if self.Nk[k] == 0:
            self.Zr = [i-1 if i>k else i for i in self.Zr]
            del self.Nk[k]
            for dim in self.dims.values():
                dim.bulk_unincorporate(k)
        self.Zr[rowid] = np.nan

    # --------------------------------------------------------------------------
    # Update schema.

    def update_cctype(self, col, cctype, hypers=None, distargs=None):
        """Update the distribution type of self.dims[col] to cctype."""
        if distargs is None:
            distargs = {}
        distargs['cctypes'] = self._unconditional_cctypes()
        distargs['ccargs'] = self._unconditional_ccargs()
        D_old = self.dims[col]
        D_new = Dim(cctype, col, hypers=hypers, distargs=distargs, rng=self.rng)
        self.unincorporate_dim(D_old)
        self.incorporate_dim(D_new)

    # --------------------------------------------------------------------------
    # Accounting

    def set_dataset(self, X):
        """Update pointer to global dataset X, see __init__ for contract."""
        self.X = X

    def reindex_dims(self):
        """Update dict(indices->dims). Invoke when global dim indices change."""
        dims = dict()
        for dim in self.dims.values():
            dims[dim.index] = dim
        self.dims = dims

    def reindex_rows(self):
        """Update row partition by deleting nans. Invoke when rowids in
        unincorporate_row are deleted from the global dataset X."""
        self.Zr = [z for z in self.Zr if not np.isnan(z)]
        assert len(self.Zr) == len(self.X)

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N):
        """Run all the transitions N times."""
        for _ in xrange(N):
            self.transition_rows()
            self.transition_alpha()
            self.transition_column_hypers()

    def transition_alpha(self):
        """Calculate CRP alpha conditionals over grid and transition."""
        logps = [gu.logp_crp_unorm(len(self.Zr), len(self.Nk), alpha)
            for alpha in self.alpha_grid]
        index = gu.log_pflip(logps, rng=self.rng)
        self.alpha = self.alpha_grid[index]

    def transition_column_hypers(self, cols=None):
        """Calculate column (dim) hyperparameter conditionals over grid and
        transition."""
        if cols is None:
            cols = self.dims.keys()
        for c in cols:
            self.dims[c].transition_hypers()

    def transition_rows(self, rows=None):
        """Compute row conditions for each cluster and transition."""
        if rows is None:
            rows = xrange(len(self.Zr))
        for rowid in rows:
            self._transition_row(rowid)

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence):
        if self._is_hypothetical(rowid):
            return self._logpdf_hypothetical(query, evidence)
        else:
            return self._logpdf_observed(rowid, query, evidence)

    def _logpdf_observed(self, rowid, query, evidence):
        # XXX Should row cluster be renegotiated based on new evidence?
        return sum([self.dims[c].logpdf(v, self.Zr[rowid]) for (c,v) in query])

    def _logpdf_hypothetical(self, query, evidence):
        # Algorithm. Partition all columns in query and evidence by views.
        # P(x1,x2|x3,x4) where (x1...x4) in the same view.
        #   = \sum_z p(x1,x2|z,x3,x4)p(z|x3,x4)     marginalization
        #   = \sum_z p(x1,x2|z)p(z|x3,x4)           conditional independence
        #   = \sum_z p(x1|z)p(x2|z)p(z|x3,x4)       conditional independence
        # Now consider p(z|x3,x4)
        #   \propto p(z)p(x3|z)p(x4|z)              Bayes rule
        # [term]           [array]
        # p(z)             logp_crp
        # p(x3|z)p(x4|z)   logp_evidence
        # p(z|x3,x4)       logp_cluster
        # p(x1|z)p(x2|z)   logp_query
        logp_crp = gu.logp_crp_fresh(self.Nk, self.Zr, self.alpha)
        logp_evidence = self._cluster_query_logps(evidence)
        logp_cluster = gu.log_normalize(np.add(logp_crp, logp_evidence))
        logp_query = self._cluster_query_logps(query)
        return logsumexp(np.add(logp_cluster, logp_query))

    def logpdf_marginal(self):
        """Compute the marginal logpdf CRP assignment and data."""
        logp_crp = gu.logp_crp(len(self.Zr), self.Nk, self.alpha)
        logp_dims = [sum(dim.logpdf_marginal()) for dim in self.dims.values()]
        return logp_crp + sum(logp_dims)

    # --------------------------------------------------------------------------
    # simulate

    def simulate(self, rowid, query, evidence, N=1):
        if self._is_hypothetical(rowid):
            return self._simulate_hypothetical(query, evidence, N)
        else:
            return self._simulate_observed(rowid, query, evidence, N)

    def _simulate_observed(self, rowid, query, evidence, N):
        # XXX Should row cluster be renegotiated based on new evidence?
        k = self.Zr[rowid]
        samples = [[self.dims[c].simulate(k) for c in query] for _ in xrange(N)]
        return np.asarray(samples)

    def _simulate_hypothetical(self, query, evidence, N, cluster=False):
        """cluster=True exposes latent cluster of each sample as extra col."""
        logp_crp = gu.logp_crp_fresh(self.Nk, self.Zr, self.alpha)
        logp_evidence = self._cluster_query_logps(evidence)
        logp_cluster = np.add(logp_crp, logp_evidence)
        ks = gu.log_pflip(logp_cluster, size=N, rng=self.rng)
        samples = [[self.dims[c].simulate(k) for c in query] for k in ks]
        return np.column_stack((samples, ks)) if cluster else np.asarray(samples)

    # --------------------------------------------------------------------------
    # Internal row transition.

    def _transition_row(self, rowid):
        # Skip unincorporated rows.
        if self.Zr[rowid] == np.nan: return
        logp_data = self._logpdf_row_gibbs(rowid, 1)
        logp_crp = gu.logp_crp_gibbs(self.Nk, self.Zr, rowid, self.alpha, 1)
        assert len(logp_data) == len(logp_crp)
        p_cluster = np.add(logp_data, logp_crp)
        z_b = gu.log_pflip(p_cluster, rng=self.rng)
        if z_b != self.Zr[rowid]:
            self.unincorporate_row(rowid)
            self.incorporate_row(rowid, z_b)
        self._check_partitions()

    def _logpdf_row_gibbs(self, rowid, m):
        """Internal use only for Gibbs transition."""
        m_aux = m-1 if self.Nk[self.Zr[rowid]]==1 else m
        return [sum([self._logpdf_gibbs(dim, rowid, k) for dim in
            self.dims.values()]) for k in xrange(len(self.Nk) + m_aux)]

    def _logpdf_gibbs(self, dim, rowid, k):
        x = self.X[rowid, dim.index]
        y = self._unconditional_values(rowids=rowid)[0] if \
            dim.is_conditional() else None
        return self._logpdf_gibbs_current(dim, x, y, k) if self.Zr[rowid] == k \
            else dim.logpdf(x, k, y=y)

    def _logpdf_gibbs_current(self, dim, x, y, k):
        dim.unincorporate(x, k, y=y)
        logp = dim.logpdf(x, k, y=y)
        dim.incorporate(x, k, y=y)
        return logp

    # --------------------------------------------------------------------------
    # Internal query utils.

    def _cluster_query_logps(self, query):
        """Returns a list of log probabilities of a query, 1 entry for each of
        the clusters in self.Nk, including a singleton."""

        def conditional_logps(c, v):
            # XXX Placeholder.
            # find known unconditionals in the joint query.
            # find missing unconditional from the joint query.
            # for each k \in K simulate missing columns.
            # compute average logpdf under the known and simulated.
            return np.zeros(len(self.Nk)+1)

        def unconditional_logps(c, x):
            return [self.dims[c].logpdf(x,k) for k in xrange(len(self.Nk)+1)]

        def column_logps(c, v):
            return conditional_logps(c,v) if self.dims[c].is_conditional() else\
                unconditional_logps(c,v)

        return np.sum([column_logps(c,v) for c,v in query], axis=0) if query \
            else np.zeros(len(self.Nk)+1)

    def _is_hypothetical(self, rowid):
        return not (0 <= rowid < len(self.Zr))

    def _unconditional_dims(self):
        return filter(lambda d: not self.dims[d].is_conditional(),
            sorted(self.dims))

    def _unconditional_values(self, rowids=None):
        unconditionals = self._unconditional_dims()
        return self.X[:,unconditionals] if rowids is None else \
            self.X[rowids,unconditionals]

    def _unconditional_cctypes(self, rowids=None):
        dims = [self.dims[i] for i in self._unconditional_dims()]
        return [d.cctype for d in dims]

    def _unconditional_ccargs(self, rowids=None):
        dims = [self.dims[i] for i in self._unconditional_dims()]
        return [d.distargs for d in dims]

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha > 0.
        assert sum(self.Nk) == len(self.Zr)
