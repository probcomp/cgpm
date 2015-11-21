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

from math import isnan

class Dim(object):
    """Holds data, model type, and hyperparameters."""

    def __init__(self, X, cc_datatype_class, index, Zr=None, n_grid=30,
            hypers=None, mode='collapsed', distargs=None):
        """Dimension constructor.

        Arguments:
        -- X: a numpy array of data.
        -- cc_datatype_class: the cc_type class for this column
        -- index: the column index (int)

        Optional arguments:
        -- Zr: partition of data to clusters. If not specified, no clusters are
        initialized.
        -- n_grid: number of hyperparameter grid bins. Default is 30.
        -- distargs: some data types require additional information, for example
        multinomial data requires the number of multinomial categories. See the
        documentation for each type for details
        """
        self.index = index
        self.X = X
        self.N = len(X)
        self.model = cc_datatype_class
        self.cctype = cc_datatype_class.cctype
        self.hypers_grids = cc_datatype_class.construct_hyper_grids(X, n_grid)
        self.distargs = distargs if distargs is not None else {}
        self.mode = mode
        self.hypers = hypers
        if hypers is None:
            self.hypers = cc_datatype_class.init_hypers(self.hypers_grids, X)

        if Zr is None:
            self.clusters = []
            self.pX = 0
        else:
            self.clusters = []
            K = max(Zr)+1
            for k in xrange(K):
                self.clusters.append(cc_datatype_class(distargs=distargs))
                self.clusters[k].set_hypers(self.hypers)

            for i in xrange(len(X)):
                k = Zr[i]
                self.clusters[k].insert_element(X[i])

    def predictive_logp(self, rowid, k):
        """Returns the predictive logp of X[rowid] in clusters[k]."""
        x = self.X[rowid]
        lp = self.clusters[k].predictive_logp(x)
        return lp

    def singleton_logp(self, rowid):
        """Returns the predictive log_p of X[rowid] in its own cluster."""
        x = self.X[rowid]
        self.aux_model = self.model(distargs=self.distargs, **self.hypers)
        lp = self.aux_model.singleton_logp(x)
        if not isinstance(lp, float):
            import ipdb; ipdb.set_trace()
        return lp

    def insert_element(self, rowid, k):
        """Insert x into clusters[k]."""
        x = self.X[rowid]
        if isnan(x):
            return
        self.clusters[k].insert_element(x)

    def remove_element(self, rowid, k):
        """Remove x from clusters[k]."""
        x = self.X[rowid]
        if isnan(x):
            return
        self.clusters[k].remove_element(x)

    def move_to_cluster(self, rowid, move_from, move_to):
        """Move X[rowid] from clusters[move_from] to clusters[move_to]."""
        x = self.X[rowid]
        self.clusters[move_from].remove_element(x)
        self.clusters[move_to].insert_element(x)

    def destroy_singleton_cluster(self, rowid, to_destroy, move_to):
        """Move X[rowid] tp clusters[move_to], destroy clusters[to_destroy]."""
        x = self.X[rowid]
        self.clusters[move_to].insert_element(x)
        del self.clusters[to_destroy]

    def create_singleton_cluster(self, rowid, current):
        """Remove X[rowid] from clusters[current] and create a new singleton
        cluster.
        """
        x = self.X[rowid]                          # get the element
        self.clusters[current].remove_element(x)   # remove from current cluster
        self.clusters.append(self.aux_model)       # create the singleton
        self.clusters[-1].insert_element(x)        # add element to new cluster

    def marginal_logp(self, k):
        """Returns the marginal log_p of clusters[k]."""
        return self.clusters[k].marginal_logp()

    def full_marginal_logp(self):
        """Returns the marginal log_p over all clusters."""
        lp = 0
        for cluster in self.clusters:
            lp += cluster.marginal_logp()

        return lp

    def transition_hypers(self):
        """Updates the hyperparameters and the component parameters."""
        for cluster in self.clusters:
            cluster.transition_params()
        self.hypers = self.model.transition_hypers(self.clusters, self.hypers,
            self.hypers_grids)

    def reassign(self, Zr):
        """Reassigns the data to new clusters according to the new
        partitioning, Zr.

        Destroys and recreates dims.
        """
        self.clusters = []
        self.Zr = Zr
        K = max(Zr) + 1

        for k in xrange(K):
            cluster = self.model(distargs=self.distargs)
            cluster.set_hypers(self.hypers)
            self.clusters.append(cluster)

        for i in xrange(self.N):
            k = Zr[i]
            self.clusters[k].insert_element(self.X[i])

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
        # FIXME: add get_suffstats() method to all types
        suffstats =[]
        # for cluster in self.clusters:
        #     suffstats.append(cluster.get_suffstats())
        return suffstats

    def plot_dist(self, ax=None):
        """Plots the predictive distribution and histogram of X."""
        self.model.plot_dist(self.X, self.clusters, distargs=self.distargs,
            ax=ax)
