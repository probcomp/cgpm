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
from math import isnan

import gpmcc.utils.general as gu
import gpmcc.utils.config as cu

class Dim(object):
    """Holds data, model type, clusters, and shared hyperparameters of
    component GPMs. Exposes the cluster of each row for model dependent
    composition."""

    def __init__(self, X, dist, index, distargs=None, Zr=None, n_grid=30,
            hypers=None, mode='collapsed'):
        """Dimension constructor.

        Arguments:
        ... X (np.array) : Array of data. Must be compatible with `dist`.
        Missing entries must be np.nan.
        ... dist (str) : Name of the DistributionGpm, see
        `gpmcc.utils.config`.
        ... index (int) : Identifier for this dim.

        Keyword Arguments:
        ... Zr (list) L Partition of data into clusters, where Zr[i] is the
        cluster index of row i. If None, no clusters created.
        ... n_grid (int) : Number of bins in the hyperparameter grid.
        """
        # Data information.
        self.N = len(X)
        self.X = X
        self.index = index

        # Model type.
        self.mode = mode
        self.model = cu.dist_class(dist)
        self.cctype = self.model.name()
        self.distargs = distargs if distargs is not None else {}

        # Hyperparams.
        self.hypers_grids = self.model.construct_hyper_grids(
            self.X[~np.isnan(X)], n_grid)
        self.hypers = hypers
        if hypers is None:
            self.hypers = dict()
            for h in self.hypers_grids:
                self.hypers[h] = np.random.choice(self.hypers_grids[h])

        # Row clusters.
        self.clusters = []
        if Zr is not None:
            K = max(Zr)+1
            for k in xrange(K):
                self.clusters.append(self.model(distargs=distargs,
                    **self.hypers))
            for i in xrange(len(X)):
                k = Zr[i]
                if not isnan(X[i]):
                    self.clusters[k].incorporate(X[i])

        # Auxiliary singleton model.
        self.aux_model = None

    def predictive_logp(self, rowid, k):
        """Returns the predictive logp of X[rowid] in clusters[k]."""
        x = self.X[rowid]
        if isnan(x):
            return 0
        lp = self.clusters[k].predictive_logp(x)
        return lp

    def singleton_logp(self, rowid):
        """Returns the predictive log_p of X[rowid] in its own cluster."""
        x = self.X[rowid]
        if isnan(x):
            return 0
        self.aux_model = self.model(distargs=self.distargs, **self.hypers)
        lp = self.aux_model.singleton_logp(x)
        return lp

    def insert_element(self, rowid, k):
        """Insert x into clusters[k]."""
        x = self.X[rowid]
        if isnan(x):
            return
        self.clusters[k].incorporate(x)

    def remove_element(self, rowid, k):
        """Remove x from clusters[k]."""
        x = self.X[rowid]
        if isnan(x):
            return
        self.clusters[k].unincorporate(x)

    def move_to_cluster(self, rowid, move_from, move_to):
        """Move X[rowid] from clusters[move_from] to clusters[move_to]."""
        x = self.X[rowid]
        if isnan(x):
            return
        self.clusters[move_from].unincorporate(x)
        self.clusters[move_to].incorporate(x)

    def destroy_singleton_cluster(self, rowid, to_destroy, move_to):
        """Move X[rowid] tp clusters[move_to], destroy clusters[to_destroy]."""
        x = self.X[rowid]
        if isnan(x):
            return
        self.clusters[move_to].incorporate(x)
        del self.clusters[to_destroy]

    def create_singleton_cluster(self, rowid, current):
        """Remove X[rowid] from clusters[current] and create a new singleton
        cluster.
        """
        x = self.X[rowid]                          # get the element
        self.clusters.append(self.aux_model)       # create the singleton
        if isnan(x):
            return
        self.clusters[current].unincorporate(x)   # remove from current cluster
        self.clusters[-1].incorporate(x)          # add element to new cluster

    def marginal_logp(self, k=None):
        """If k is not None, teturns the marginal log_p of clusters[k].
        Otherwise returns the sum of marginal log_p over all clusters."""
        if k is not None:
            return self.clusters[k].marginal_logp()
        return sum(cluster.marginal_logp() for cluster in self.clusters)

    def transition_hypers(self):
        """Updates the hyperparameters and the component parameters."""
        # Transition component parameters.
        for cluster in self.clusters:
            cluster.transition_params()
        # Transition hyperparameters.
        targets = self.hypers.keys()
        np.random.shuffle(targets)
        for target in targets:
            logps = self.calc_hyper_proposal_logps(target)
            proposal = gu.log_pflip(logps)
            self.hypers[target] = self.hypers_grids[target][proposal]
        # Update the clusters.
        for cluster in self.clusters:
            cluster.set_hypers(self.hypers)

    def calc_hyper_proposal_logps(self, target):
        """Computes the marginal likelihood (over all clusters) for each
        hyperparameter value in self.hypers_grids[target].
        p(h|X) \prop p(h)p(X|h)
        """
        logps = []
        hypers = self.hypers.copy()
        for g in self.hypers_grids[target]:
            hypers[target] = g
            logp = 0
            for cluster in self.clusters:
                cluster.set_hypers(hypers)
                logp += cluster.marginal_logp()
                cluster.set_hypers(self.hypers)
            logps.append(logp)
        return logps

    def reassign(self, Zr):
        """Reassigns the data to new clusters according to the new
        partitioning, Zr.

        Destroys and recreates clusters.
        """
        self.clusters = []
        self.Zr = Zr
        K = max(Zr) + 1

        for k in xrange(K):
            cluster = self.model(distargs=self.distargs, **self.hypers)
            self.clusters.append(cluster)

        for i in xrange(self.N):
            k = Zr[i]
            if not isnan(self.X[i]):
                self.clusters[k].incorporate(self.X[i])

        for cluster in self.clusters:
            cluster.transition_params()

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

    def get_suffstats(self):
        suffstats =[]
        for cluster in self.clusters:
            suffstats.append(cluster.get_suffstats())
        return suffstats

    def plot_dist(self, Y=None, ax=None):
        """Plots the predictive distribution and histogram of X."""
        self.model.plot_dist(self.X[~np.isnan(self.X)], self.clusters,
            ax=ax, Y=Y, hist=False)
