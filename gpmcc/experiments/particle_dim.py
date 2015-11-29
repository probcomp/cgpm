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

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import logsumexp

from gpmcc.utils import config as cu
from gpmcc.utils import general as gu

class ParticleDim(object):
    """Holds data, model type, and hyperparameters."""

    def __init__(self, dist, distargs=None, n_grid=30):
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
        # Model type.
        self.model = cu.dist_class(dist)
        self.cctype = self.model.cctype
        self.distargs = distargs if distargs is not None else {}

        # Distribution hyperparams.
        self.n_grid = n_grid
        self.hypers_grids = dict()
        self.hypers = dict()

        # CRP hyperparams.
        self.alpha = 1
        self.alpha_grid = []

        # Initial state variables.
        self.weight = 0
        self.clusters = []
        self.Xobs = np.asarray([])
        self.Nobs = 0
        self.Zr = np.asarray([])

    def particle_initialize(self, x):
        """Clears entire state to a single observation."""
        # Initialize the SMC with first observation into cluster 0.
        self.clusters = [self.model(distargs=self.distargs, **self.hypers)]
        self.Nobs = 1
        self.Xobs = np.asarray([x])
        self.Zr = np.asarray([0])
        self.insert_element(0, self.Zr[0])
        self.weight = self.clusters[0].marginal_logp()
        # Hyperparameters.
        self.update_grids(self.Xobs)
        self.hypers = self.model.init_hypers(self.hypers_grids, self.Xobs)

    def particle_learn(self, X, n_gibbs=1, plot=False, progress=False):
        """Iteratively applies particle learning on the observations in X."""
        if self.Nobs == 0:
            self.particle_initialize(X[0])
            X = X[1:]
        if plot:
            plt.ion(); plt.show()
            _, ax = plt.subplots()
            self.plot_dist(ax=ax)
            plt.draw()
            plt.pause(3)
        for i, x in enumerate(X):
            if progress:
                # Monitor progress.
                percentage = float(i+1) / len(X)
                progress = ' ' * 30
                fill = int(percentage * len(progress))
                progress = '[' + '=' * fill + progress[fill:] + ']'
                print '{} {:1.2f}%\r'.format(progress, 100 * percentage),
                sys.stdout.flush()
            # Include the new datapoint.
            self.incorporate(x)
            self.weight += self.transition_rows(target_rows=[self.Nobs-1])
            for _ in xrange(n_gibbs):
                self.gibbs_transition()
                # Plot?
                if plot:
                    ax.clear()
                    self.plot_dist(ax=ax)
                    plt.draw()
                    plt.pause(3)
        return self.weight

    def gibbs_transition(self):
        # Gibbs transition.
        self.transition_rows(target_rows=range(self.Nobs))
        self.transition_hypers()
        self.transition_alpha()


    def incorporate(self, x):
        """Incorporates a new observation in particle learning."""
        # Arbitrarily assign to cluster 0, not consequential.
        self.Nobs += 1
        self.Xobs = np.append(self.Xobs, x)
        assert self.Nobs == len(self.Xobs)
        rowid = self.Nobs - 1
        self.insert_element(rowid, 0)
        self.Zr = np.append(self.Zr, 0)
        self.update_grids(self.Xobs)

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
        # If move_from now singleton, destroy.
        if self.cluster_count(move_from) == 0:
            zminus = np.nonzero(self.Zr > move_from)
            self.Zr[zminus] -= 1
            del self.clusters[move_from]

    def create_singleton_cluster(self, rowid, current):
        """Remove X[rowid] from clusters[current] and create a new singleton
        cluster.
        """
        self.clusters.append(self.aux_model)
        self.Zr[rowid] = len(self.clusters) - 1
        self.remove_element(rowid, current)
        self.insert_element(rowid, self.Zr[rowid])

    def cluster_count(self, k):
        return sum(j == k for j in self.Zr)

    def cluster_counts(self):
        counts = [self.cluster_count(j) for j in xrange(len(self.clusters))]
        if sum(counts) != self.Nobs:
            import ipdb; ipdb.set_trace()
        return counts

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

    def marginal_logp(self, k):
        """Returns the marginal log_p of clusters[k]."""
        return self.clusters[k].marginal_logp()

    def full_marginal_logp(self):
        """Returns the marginal log_p over all clusters."""
        return sum(cluster.marginal_logp() for cluster in self.clusters)

    def update_grids(self, X):
        self.hypers_grids = self.model.construct_hyper_grids(X,
            self.n_grid)
        self.alpha_grid = gu.log_linspace(1.0/len(X), len(X),
            self.n_grid)

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

    def transition_rows(self, target_rows=None):
        """Reassign rows to categories.
        Optional arguments:
        -- target_rows: A list of rows to reassign. If not specified, reassigns
        every row in self.Nobs.
        Returns:
        -- particle weight.
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
                if z_b == len(self.clusters):
                    self.create_singleton_cluster(rowid, z_a)
                else:
                    self.move_to_cluster(rowid, z_a, z_b)

            weight += logsumexp(p_cluster)

        return weight

    def plot_dist(self, ax=None):
        """Plots the predictive distribution and histogram of X."""
        self.model.plot_dist(self.Xobs, self.clusters, distargs=self.distargs,
            ax=ax, hist=False)
