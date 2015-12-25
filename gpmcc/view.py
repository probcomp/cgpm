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

from math import log

import numpy as np
import gpmcc.utils.general as gu

class View(object):
    """View. A collection of Dim."""

    def __init__(self, X, dims, alpha=None, Zr=None, n_grid=30):
        """View constructor.

        Arguments:
        ... X (np.ndarray) : Global dataset NxD. The invariant is that
        the data from dim.index should be in X[:,dim.index].
        ... dims (list<dim>) : A list of Dim objects in this View.

        Keyword Arguments:
        ... alpha (float): CRP concentration parameter. If None, selected
        from grid uniformly at random.
        ... Zr (list<int>): Starting partiton of rows to categories.
        If None, is intialized from CRP(alpha)
        ... n_grid (int): Number of grid points in hyperparameter grids.
        """
        # Dataset.
        self.X = X
        self.N = len(X)
        for dim in dims:
            assert self.N == dim.N

        # Generate alpha.
        self.alpha_grid = gu.log_linspace(1./self.N, self.N, n_grid)
        if alpha is None:
            alpha = np.random.choice(self.alpha_grid)
        assert alpha > 0.
        self.alpha = alpha

        # Generate row partition.
        if Zr is None:
            Zr, Nk, _ = gu.crp_gen(self.N, alpha)
        else:
            Nk = gu.bincount(Zr)
        assert len(Zr) == self.N
        assert sum(Nk) == self.N
        self.Zr = np.array(Zr)
        self.Nk = Nk

        # Initialize the dimensions.
        self.dims = dict()
        for dim in dims:
            dim.reassign(X[:,dim.index], Zr)
            self.dims[dim.index] = dim

    def transition(self, N):
        """Run all the transitions N times."""
        for _ in xrange(N):
            self.transition_Zr()
            self.transition_alpha()
            self.transition_column_hypers()

    def transition_alpha(self):
        """Calculate CRP alpha conditionals over grid and transition."""
        logps = np.zeros(len(self.alpha_grid))
        for i in range(len(self.alpha_grid)):
            alpha = self.alpha_grid[i]
            logps[i] = gu.unorm_lcrp_post(alpha, self.N, len(self.Nk),
                lambda x: 0)
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def transition_column_hypers(self):
        """Calculate column (dim) hyperparameter conditionals over grid and
        transition.
        """
        for dim in self.dims.values():
            dim.transition_hypers()

    def transition_Zr(self, target_rows=None, N=1):
        """Transition row assignments.

        Keyword arguments:
        ... target_rows (list<int>): List of rows to reassign.
        If not specified, reassigns every row.
        ... N (int): Number of times to transition.
        """
        for _ in range(N):
            self.transition_rows(target_rows=target_rows)

    def transition_rows(self, target_rows=None):
        """Reassign rows to clusters.
        Keyword Arguments:
        ... target_rows (list<int>): Rows to reassign. If None, transitions
        every row.
        """
        if target_rows is None:
            target_rows = [i for i in xrange(self.N)]

        for rowid in target_rows:
            # Get current assignment z_a.
            z_a = self.Zr[rowid]
            is_singleton = (self.Nk[z_a] == 1)

            # Get CRP probabilities.
            p_crp = list(self.Nk)
            if is_singleton:
                # If z_a is singleton do not consider a new singleton.
                p_crp[z_a] = self.alpha
            else:
                # Decrement current cluster count.
                p_crp[z_a] -= 1
                # Append to the CRP an alpha for singleton.
                p_crp.append(self.alpha)

            # Log-normalize p_crp.
            p_crp = np.log(np.array(p_crp))
            p_crp = gu.log_normalize(p_crp)

            # Calculate probability of rowid in each cluster k \in K.
            p_cluster = []
            for k in xrange(len(self.Nk)):
                # If k == z_a then predictive_logp will remove rowid's
                # suffstats and reuse parameters.
                lp = self.row_predictive_logp(rowid, k) + p_crp[k]
                p_cluster.append(lp)

            # Propose singleton.
            if not is_singleton:
                # Using len(self.Nk) will resample parameters.
                lp = self.row_predictive_logp(rowid, len(self.Nk)) + \
                    p_crp[-1]
                p_cluster.append(lp)

            # Draw new assignment, z_b
            z_b = gu.log_pflip(p_cluster)

            # Migrate the row.
            self._move_row_to_cluster(rowid, z_a, z_b)

    def row_predictive_logp(self, rowid, k):
        """Get the predictive log_p of rowid being in cluster k. If k
        is existing (less than len(self.Nk)) then the predictive is taken.
        If k is new (equal to len(self.Nk)) then new parameters
        are sampled for the predictive."""
        assert k <= len(self.Nk)
        logp = 0
        for dim in self.dims.values():
            x = self.X[rowid, dim.index]
            # If rowid already in cluster k, need to unincorporate first.
            if self.Zr[rowid] == k:
                dim.unincorporate(x, k)
                logp += dim.predictive_logp(x, k)
                dim.incorporate(x, k)
            else:
                logp += dim.predictive_logp(x, k)
        return logp

    def insert_dim(self, dim):
        self.dims[dim.index] = dim
        if not np.allclose(dim._Zr_last, self.Zr):
            dim.reassign(self.X[:, dim.index], self.Zr)

    def remove_dim(self, dim_index):
        del self.dims[dim_index]

    def _move_row_to_cluster(self, rowid, move_from, move_to):
        """Move rowid from cluster move_from to move_to. If move_to
        is len(self.Nk) a new cluster will be created."""
        assert move_from < len(self.Nk) and move_to <= len(self.Nk)

        # Do nothing.
        if move_from == move_to:
            return

        # Notify dims.
        for dim in self.dims.values():
            dim.unincorporate(self.X[rowid, dim.index], move_from)
            dim.incorporate(self.X[rowid, dim.index], move_to)

        # Update partition and move_from counts.
        self.Zr[rowid] = move_to
        self.Nk[move_from] -= 1

        # If move_to new cluster, extend Nk.
        if move_to == len(self.Nk):
            self.Nk.append(0)
            # Never create a singleton cluster from another singleton.
            assert self.Nk[move_from] != 0

        # Update move_to counts.
        self.Nk[move_to] += 1

        # If move_from is now empty, delete and update cluster ids.
        if self.Nk[move_from] == 0:
            assert move_to != len(self.Nk)
            self.Zr[np.nonzero(self.Zr > move_from)] -= 1
            for dim in self.dims.values():
                dim.destroy_cluster(move_from)
            del self.Nk[move_from]
