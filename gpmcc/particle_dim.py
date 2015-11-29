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

import numpy as np
from scipy.misc import logsumexp

from gpmcc.utils import config as cu
from gpmcc.utils import general as gu

class ParticleDim(object):
    """Holds data, model type, and hyperparameters."""

    def __init__(self, X, dist, distargs=None, n_grid=30, hypers=None,
            mode='collapsed'):
        """Dimension constructor.

        Arguments:
        -- X: a numpy array of data.
        -- dist: The name of the distribution.

        Optional arguments:
        -- Zr: partition of data to clusters. If not specified, no clusters are
        initialized.
        -- n_grid: number of hyperparameter grid bins. Default is 30.
        -- distargs: some data types require additional information, for example
        multinomial data requires the number of multinomial categories. See the
        documentation for each type for details
        """
        # Data information.
        self.X = X

        # Model type.
        self.mode = mode
        self.model = cu.dist_class(dist)
        self.cctype = self.model.cctype
        self.distargs = distargs if distargs is not None else {}

        # Hyperparams.
        self.hypers_grids = self.model.construct_hyper_grids(self.X, n_grid)
        self.hypers = hypers
        if hypers is None:
            self.hypers = self.model.init_hypers(self.hypers_grids, self.X)

        # CRP
        self.alpha = 1
        self.alpha_grid = gu.log_linspace(1.0/len(self.X), len(self.X), n_grid)

    def particle_initialize(self):
        # Initialize the SMC with first observation into cluster 0.
        self.clusters = [self.model(distargs=self.distargs, **self.hypers)]
        self.Nobs = 1
        self.Xobs = self.X[:self.Nobs]
        self.Zr = [0]
        self.insert_element(0, self.Zr[0])
        self.weight = self.clusters[0].marginal_logp()

    def particle_learn(self):
        self.particle_initialize()
        for j in xrange(1, len(self.X)):
            self.weight += self.incorporate_observation(j)
            self.transition_rows(target_rows=range(j+1))
            self.transition_hypers()
            self.transition_alpha()
        return self.weight

    def incorporate_observation(self, rowid):
        # Arbitrarily assign to cluster 0, not consequential.
        self.Nobs += 1
        self.Xobs = self.X[:self.Nobs]
        self.insert_element(rowid, 0)
        self.Zr = np.append(self.Zr, 0)
        # Transition the row.
        weight = self.transition_rows(target_rows=[rowid])
        return weight

    def predictive_logp(self, rowid, k):
        """Returns the predictive logp of X[rowid] in clusters[k]."""
        x = self.Xobs[rowid]
        if self.Zr[rowid] == k:
            self.remove_element(rowid, k)
            lp = self.clusters[k].predictive_logp(x)
            self.insert_element(rowid, k)
        else:
            lp = self.clusters[k].predictive_logp(x)
        return lp

    def singleton_logp(self, rowid):
        """Returns the predictive log_p of X[rowid] in its own cluster."""
        x = self.Xobs[rowid]
        self.aux_model = self.model(distargs=self.distargs, **self.hypers)
        lp = self.aux_model.singleton_logp(x)
        return lp

    def insert_element(self, rowid, k):
        """Insert x into clusters[k]."""
        x = self.Xobs[rowid]
        self.clusters[k].insert_element(x)

    def remove_element(self, rowid, k):
        """Remove x from clusters[k]."""
        x = self.Xobs[rowid]
        self.clusters[k].remove_element(x)

    def move_to_cluster(self, rowid, move_from, move_to):
        """Move X[rowid] from clusters[move_from] to clusters[move_to]."""
        self.remove_element(rowid, move_from)
        self.insert_element(rowid, move_to)
        self.Zr[rowid] = move_to

    def destroy_singleton_cluster(self, rowid, to_destroy, move_to):
        """Move X[rowid] to clusters[move_to], destroy clusters[to_destroy]."""
        self.insert_element(rowid, move_to)
        self.Zr[rowid] = move_to
        zminus = np.nonzero(self.Zr>to_destroy)
        self.Zr[zminus] -= 1
        del self.clusters[to_destroy]

    def create_singleton_cluster(self, rowid, current):
        """Remove X[rowid] from clusters[current] and create a new singleton
        cluster.
        """
        self.clusters.append(self.aux_model)
        self.Zr[rowid] = len(self.clusters) - 1
        self.remove_element(rowid, current)
        self.insert_element(rowid, self.Zr[rowid])

    def marginal_logp(self, k):
        """Returns the marginal log_p of clusters[k]."""
        return self.clusters[k].marginal_logp()

    def full_marginal_logp(self):
        """Returns the marginal log_p over all clusters."""
        return sum(cluster.marginal_logp() for cluster in self.clusters)

    def clear_data(self):
        """Removes all data from the clusters. Cleans suffstats."""
        K = len(self.clusters)
        for k in xrange(K):
            cluster = self.model(distargs=self.distargs)
            cluster.set_hypers(self.hypers)
            self.clusters[k] = cluster

    def update_prior_grids(self):
        n_grid = len(self.hypers_grids.values()[0])
        hypers_grids = self.model.construct_hyper_grids(self.X, n_grid)
        hypers = self.model.init_hypers(hypers_grids, self.X)
        self.hypers_grids = hypers_grids
        self.hypers = hypers
        for cluster in self.clusters:
            cluster.set_hypers(hypers)

    def transition_hypers(self):
        """Updates the hyperparameters and the component parameters."""
        for cluster in self.clusters:
            cluster.transition_params()
        self.hypers = self.model.transition_hypers(self.clusters, self.hypers,
            self.hypers_grids)

    def transition_alpha(self):
        """Calculate CRP alpha conditionals over grid and transition."""
        K = len(self.clusters)
        logps = np.zeros(len(self.alpha_grid))
        for i in range(len(self.alpha_grid)):
            alpha = self.alpha_grid[i]
            logps[i] = gu.unorm_lcrp_post(alpha, self.Nobs, K, lambda x: 0)
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def cluster_count(self, k):
        return sum(j == k for j in self.Zr)

    def cluster_counts(self):
        counts = [self.cluster_count(j) for j in xrange(len(self.clusters))]
        if sum(counts) != self.Nobs:
            import ipdb; ipdb.set_trace()
        return counts

    def transition_rows(self, target_rows=None):
        """Reassign rows to categories.
        Optional arguments:
        -- target_rows: a list of rows to reassign. If not specified, reassigns
        every row in self.Nobs.
        """
        weight = 0
        log_alpha = np.log(self.alpha)

        if target_rows is None:
            target_rows = [i for i in xrange(self.Nobs)]

        for rowid in target_rows:
            # Get current assignment z_a.
            z_a = self.Zr[rowid]
            is_singleton = (self.cluster_count(z_a) == 1)

            # Get CRP probabilities.
            p_crp = self.cluster_counts()
            if is_singleton:
                # If z_a is singleton do not consider a new singleton.
                p_crp[z_a] = self.alpha
            else:
                # Remove rowid from the z_a cluster count.
                p_crp[z_a] -= 1

            # Take log of the CRP probabilities.
            p_crp = np.log(np.array(p_crp))

            # Calculate probability of each row in each category, k \in K.
            p_cluster = []
            for k in xrange(len(self.clusters)):
                if k == z_a and is_singleton:
                    lp = self.singleton_logp(rowid) + p_crp[k]
                else:
                    lp = self.predictive_logp(rowid, k) + p_crp[k]
                p_cluster.append(lp)

            # Propose singleton.
            if not is_singleton:
                lp = self.singleton_logp(rowid) + log_alpha
                p_cluster.append(lp)

            # Draw new assignment, z_b
            z_b = gu.log_pflip(p_cluster)

            if z_a != z_b:
                if is_singleton:
                    self.destroy_singleton_cluster(rowid, z_a, z_b)
                elif z_b == len(self.clusters):
                    self.create_singleton_cluster(rowid, z_a)
                else:
                    self.move_to_cluster(rowid, z_a, z_b)

            weight += logsumexp(p_cluster)
            # self._check_partitions()

        return weight

    def plot_dist(self, ax=None):
        """Plots the predictive distribution and histogram of X."""
        self.model.plot_dist(self.Xobs, self.clusters, distargs=self.distargs,
            ax=ax)
