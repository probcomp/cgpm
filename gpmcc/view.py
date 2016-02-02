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

import numpy as np
import gpmcc.utils.general as gu

class View(object):
    """View, a collection of Dim and their row mixtures."""

    def __init__(self, X, dims, alpha=None, Zr=None, n_grid=30):
        """View constructor.

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
        Zr : list<int>
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
            Zr, Nk, _ = gu.simulate_crp(len(self.X), alpha)
        else:
            Nk = list(np.bincount(Zr))
        self.Zr = list(Zr)
        self.Nk = Nk

        # Initialize the dimensions.
        self.dims = dict()
        for dim in dims:
            dim.reassign(X[:,dim.index], Zr)
            self.dims[dim.index] = dim

        # self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, dim, reassign=True):
        """Incorporate the dim into this view. If reassign is False, the row
        partition of dim must match self.Zr already.
        """
        self.dims[dim.index] = dim
        if reassign:
            dim.reassign(self.X[:, dim.index], self.Zr)
        return dim.logpdf_marginal()

    def unincorporate_dim(self, dim):
        """Remove dim from this view (does not modify dim)."""
        del self.dims[dim.index]
        return dim.logpdf_marginal()

    def incorporate_row(self, rowid, k=None):
        """Incorporate rowid from the global dataset X into the view. Use
        set_dataset to update X if it has new rowids.

        Parameters
        ----------
        rowid : int
            The rowid in dataset X to be incorporated. If rowid is already
            incorporated an error will be thrown.
        k : int, optional
            Index of the cluster to assign the row. If unspecified, will be
            sampled. If 0 <= k < len(view.Nk) then will insert
            into an existing cluser. If k = len(state.Nv) a singleton cluster
            will be created with uncollapsed parameters from the prior.
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
        self.transition_rows(target_rows=transition)

    def unincorporate_row(self, rowid):
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
                dim.destroy_cluster(k)
        self.Zr[rowid] = np.nan

    # --------------------------------------------------------------------------
    # Accounting

    def set_dataset(self, X):
        """Update the pointer to the global dataset X. The invariant is that
        the data for dim.index should be in column X[:,dim.index] and the data
        for rowid should X[rowid,:].
        """
        self.X = X

    def reindex_dims(self):
        """Update the dict mapping dim indices to dims."""
        dims = dict()
        for dim in self.dims.values():
            dims[dim.index] = dim
        self.dims = dims

    def reindex_rows(self):
        """Update row partition by dropping nan values."""
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

    def transition_column_hypers(self, target_cols=None):
        """Calculate column (dim) hyperparameter conditionals over grid and
        transition.
        """
        if target_cols is None:
            target_cols = self.dims.keys()
        for col in target_cols:
            self.dims[col].transition_hypers()

    def transition_rows(self, target_rows=None):
        """Transition the row partitioning. target_rows is an optional list
        of rows to transition.
        """
        if target_rows is None:
            target_rows = xrange(len(self.Zr))
        for rowid in target_rows:
            self._transition_row(rowid)

    # --------------------------------------------------------------------------
    # Internal

    def _logpdf_row(self, rowid, k):
        """Get the predictive log_p of rowid being in cluster k.
        If k is existing (less than len(self.Nk)) then the predictive is taken.
        If k is new (equal to len(self.Nk)) then new parameters are sampled for
        the predictive."""
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
            # Using len(self.Nk) will resample parameters.
            lp = self._logpdf_row(rowid, len(self.Nk)) + p_crp[-1]
            p_cluster.append(lp)

        # Draw new assignment, z_b
        z_b = gu.log_pflip(p_cluster)

        # Migrate the row.
        # self._move_row_to_cluster(rowid, z_a, z_b)
        if z_a != z_b:
            self.unincorporate_row(rowid)
            self.incorporate_row(rowid, z_b)

        self._check_partitions()

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha > 0.
        assert sum(self.Nk) == len(self.Zr)
