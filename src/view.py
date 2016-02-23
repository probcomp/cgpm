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

from math import log
from scipy.misc import logsumexp

import numpy as np
import gpmcc.utils.general as gu

class View(object):
    """View, a collection of Dim and their row mixtures."""

    def __init__(self, X, dims, alpha=None, Zr=None, n_grid=30):
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
        n_grid : int
            Number of grid points in hyperparameter grids.
        """
        # Dataset.
        self.X = X

        # Generate alpha.
        self.alpha_grid = gu.log_linspace(1./len(self.X), len(self.X), n_grid)
        if alpha is None:
            alpha = np.random.choice(self.alpha_grid)
        self.alpha = alpha

        # Generate row partition.
        if Zr is None:
            Zr = gu.simulate_crp(len(self.X), alpha)
        self.Zr = list(Zr)
        self.Nk = list(np.bincount(Zr))

        # Initialize the dimensions.
        self.dims = dict()
        for dim in dims:
            self.incorporate_dim(dim)

        # self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, dim, reassign=True):
        """Incorporate the dim into this View. If reassign is False, the row
        partition of dim should match self.Zr already."""
        self.dims[dim.index] = dim
        if reassign:
            dim.bulk_incorporate(self.X[:, dim.index], self.Zr)
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
        logps = [gu.logp_crp_unorm(len(self.Zr), len(self.Nk), alpha) for
            alpha in self.alpha_grid]
        index = gu.log_pflip(logps)
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
        # XXX Ignores evidence. Should the row cluster be renegotiated based on
        # new evidence?
        logpdf = 0
        for (col, val) in query:
            k = self.Zr[rowid]
            logpdf += self.dims[col].logpdf(val, k)
        return logpdf

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

        logp_crp = self._compute_cluster_crp_logps()
        logp_evidence = np.zeros(len(logp_crp))
        for (col, val) in evidence:
            logp_evidence += self._compute_cluster_data_logps(col, val)
        logp_cluster = gu.log_normalize(logp_crp+logp_evidence)

        # Query densities.
        logp_query = np.zeros(len(logp_crp))
        for (col, val) in query:
            logp_query += self._compute_cluster_data_logps(col, val)

        return logsumexp(logp_query+logp_cluster)

    def logpdf_marginal(self):
        """Compute the marginal logpdf of data and CRP assignment."""
        return gu.logp_crp(len(self.Zr), self.Nk, self.alpha) + \
            sum(sum(dim.logpdf_marginal()) for dim in self.dims.values())

    # --------------------------------------------------------------------------
    # simulate

    def simulate(self, rowid, query, evidence, N=1):
        if self._is_hypothetical(rowid):
            return self._simulate_hypothetical(query, evidence, N)
        else:
            return self._simulate_observed(rowid, query, evidence, N)

    def _simulate_observed(self, rowid, query, evidence, N):
        # XXX Ignores evidence. Should the row cluster be renegotiated based on
        # new evidence?
        samples = []
        for _ in xrange(N):
            draw = []
            for col in query:
                k = self.Zr[rowid]
                x = self.dims[col].simulate(k)
                draw.append(x)
            samples.append(draw)
        return np.asarray(samples)

    def _simulate_hypothetical(self, query, evidence, N):
        # CRP densities.
        logp_crp = self._compute_cluster_crp_logps()

        # Evidence densities.
        logp_evidence = np.zeros(len(logp_crp))
        for (col, val) in evidence:
            logp_evidence += self._compute_cluster_data_logps(col, val)

        cluster_logps = gu.log_normalize(logp_crp+logp_evidence)

        samples = []
        for _ in xrange(N):
            draw = []
            k = gu.log_pflip(cluster_logps)
            for col in query:
                x = self.dims[col].simulate(k)
                draw.append(x)
            samples.append(draw)

        return np.asarray(samples)

    # --------------------------------------------------------------------------
    # Internal

    def _transition_row(self, rowid):
        # Skip unincorporated rows.
        if self.Zr[rowid] == np.nan:
            return

        # Get current assignment z_a.
        z_a = self.Zr[rowid]
        is_singleton = (self.Nk[z_a] == 1)

        # Get CRP probabilities.
        p_crp = list(self.Nk)
        if is_singleton:
            p_crp[z_a] = self.alpha
        else:
            p_crp[z_a] -= 1
            p_crp.append(self.alpha)
        p_crp = gu.log_normalize(np.log(p_crp))

        # Calculate probability of rowid in each cluster k \in K.
        p_cluster = []
        for k in xrange(len(self.Nk)):
            lp = self._logpdf_row(rowid, k) + p_crp[k]
            p_cluster.append(lp)

        # Propose singleton.
        if not is_singleton:
            # Using len(self.Nk) will compute singleton.
            lp = self._logpdf_row(rowid, len(self.Nk)) + p_crp[-1]
            p_cluster.append(lp)

        # Draw new assignment, z_b
        z_b = gu.log_pflip(p_cluster)

        # Migrate the row.
        if z_a != z_b:
            self.unincorporate_row(rowid)
            self.incorporate_row(rowid, z_b)

        self._check_partitions()

    def _logpdf_row(self, rowid, k):
        """Internal use only for Gibbs. Compute logpdf(X[rowid]|cluster k).
        If k < len(self.Nk), predictive is taken. If k == len(self.Nk), new
        parameters are sampled."""
        assert k <= len(self.Nk)
        logp = 0
        for dim in self.dims.values():
            x = self.X[rowid, dim.index]
            if self.Zr[rowid] == k:
                dim.unincorporate(x, k)
                logp += dim.logpdf(x, k)
                dim.incorporate(x, k)
            else:
                logp += dim.logpdf(x, k)
        return logp

    def _compute_cluster_crp_logps(self):
        """Returns a list of log probabilities that a new row joins each of the
        clusters in self.views[view], including a singleton."""
        log_crp_numer = np.log(self.Nk + [self.alpha])
        logp_crp_denom = log(len(self.Zr) + self.alpha)
        return log_crp_numer - logp_crp_denom

    def _compute_cluster_data_logps(self, col, x):
        """Returns a list of log probabilities that a new row for self.dims[col]
        obtains value x for each of the clusters in self.Zr[col], including a
        singleton."""
        return [self.dims[col].logpdf(x,k) for k in
            xrange(len(self.dims[col].clusters)+1)]

    def _is_hypothetical(self, rowid):
        return not 0 <= rowid < len(self.Zr)

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha > 0.
        assert sum(self.Nk) == len(self.Zr)
