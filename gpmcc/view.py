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

    def __init__(self, dims, alpha=None, Zr=None, n_grid=30):
        """View constructor.

        Arguments:
        ... dims (list<dim>): A list of Dim objects in this View.

        Keyword Arguments:
        ... alpha (float): CRP concentration parameter. If None, selected
        from grid uniformly at random.
        ... Zr (list<int>): Starting partiton of rows to categories.
        If None, is intialized from CRP(alpha)
        ... n_grid (int): Number of grid points in hyperparameter grids.
        """
        N = dims[0].N
        self.N = N

        # Generate alpha.
        self.alpha_grid = gu.log_linspace(1. / self.N, self.N, n_grid)

        if alpha is None:
            alpha = np.random.choice(self.alpha_grid)
        else:
            assert alpha > 0.

        if Zr is None:
            Zr, Nk, K = gu.crp_gen(N, alpha)
        else:
            assert len(Zr) == dims[0].X.shape[0]
            Nk = gu.bincount(Zr)
            K = len(Nk)

        assert sum(Nk) == N
        assert K == len(Nk)

        self.dims = dict()
        for dim in dims:
            dim.reassign(Zr)
            self.dims[dim.index] = dim

        self.alpha = alpha
        self.Zr = np.array(Zr)
        self.K = K
        self.Nk = Nk

    def transition_rows(self, target_rows=None):
        """Reassign rows to clusters.

        Keyword Arguments:
        ... target_rows (list<int>): Rows to reassign. If None, transitions
        every row.
        """
        log_alpha = log(self.alpha)

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
            for k in xrange(self.K):
                if k == z_a and is_singleton:
                    lp = self.row_singleton_logp(rowid) + p_crp[k]
                else:
                    lp = self.row_predictive_logp(rowid, k) + p_crp[k]
                p_cluster.append(lp)

            # Propose singleton.
            if not is_singleton:
                lp = self.row_singleton_logp(rowid) + p_crp[-1]
                p_cluster.append(lp)

            # Draw new assignment, z_b
            z_b = gu.log_pflip(p_cluster)

            if z_a != z_b:
                if is_singleton:
                    self.destroy_singleton_cluster(rowid, z_a, z_b)
                elif z_b == self.K:
                    self.create_singleton_cluster(rowid, z_a)
                else:
                    self.move_row_to_cluster(rowid, z_a, z_b)

            # self._check_partitions()

    def transition(self, N):
        """Run all the transitions."""
        for _ in xrange(N):
            self.transition_Z()
            self.transition_alpha()
            self.transition_column_hypers()

    def transition_alpha(self):
        """Calculate CRP alpha conditionals over grid and transition."""
        logps = np.zeros(len(self.alpha_grid))
        for i in range(len(self.alpha_grid)):
            alpha = self.alpha_grid[i]
            logps[i] = gu.unorm_lcrp_post(alpha, self.N, self.K, lambda x: 0)
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def transition_column_hypers(self):
        """Calculate column (Dim) hyperparameter conditionals over grid and
        transition.
        """
        for dim in self.dims.values():
            dim.transition_hypers()

    def transition_Z(self, target_rows=None, N=1):
        """Transition row assignment.
        Optional arguments:
        -- target_rows: a list of rows to reassign. If not specified, reassigns
        every row
        -- N: number of times to transition (defualt: 1)
        """
        for _ in range(N):
            self.transition_rows(target_rows=target_rows)

    def row_predictive_logp(self, rowid, cluster):
        """Get the predictive log_p of rowid being in cluster."""
        return sum(dim.predictive_logp(rowid, cluster) for dim in
            self.dims.values())

    def row_singleton_logp(self, rowid):
        """Get the predictive log_p of rowid being a singleton cluster."""
        return sum(dim.singleton_logp(rowid) for dim in self.dims.values())

    def destroy_singleton_cluster(self, rowid, to_destroy, move_to):
        self.Zr[rowid] = move_to
        zminus = np.nonzero(self.Zr>to_destroy)
        self.Zr[zminus] -= 1
        for dim in self.dims.values():
            dim.destroy_singleton_cluster(rowid, to_destroy, move_to)
        self.Nk[move_to] += 1
        del self.Nk[to_destroy]
        self.K -= 1

    def create_singleton_cluster(self, rowid, current):
        self.Zr[rowid] = self.K
        self.K += 1
        self.Nk[current] -= 1
        self.Nk.append(1)
        for dim in self.dims.values():
            dim.create_singleton_cluster(rowid, current)

    def move_row_to_cluster(self, rowid, move_from, move_to):
        self.Zr[rowid] = move_to
        self.Nk[move_from] -= 1
        self.Nk[move_to] += 1
        for dim in self.dims.values():
            dim.move_to_cluster(rowid, move_from, move_to)

    def insert_dim(self, dim):
        if not np.allclose(dim.Zr, self.Zr):
            dim.reassign(self.Zr)
        self.dims[dim.index] = dim

    def remove_dim(self, dim_index):
        del self.dims[dim_index]

    def clear_data(self):
        for dim in self.dims.values():
            dim.clear_data()

    def _check_partitions(self):
        # For debugging only.
        # The counts for all clusters are accounted for.
        assert len(self.Nk) == self.K
        # All rows must be accounted for in the clustering.
        assert sum(self.Nk) == self.N
        zs = sorted(list(set(self.Zr)))
        for j in xrange(self.K):
          assert zs[j] == j
          for dim in self.dims.keys():
              assert self.dims[dim].clusters[j].N == self.Nk[j]
