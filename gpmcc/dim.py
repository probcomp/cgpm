# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
#
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

    def __init__(self, X, cc_datatype_class, index, Z=None, n_grid=30,
            distargs=None):
        """Dimension constructor.

        Arguments:
        -- X: a numpy array of data.
        -- cc_datatype_class: the cc_type class for this column
        -- index: the column index (int)

        Optional arguments:
        -- Z: partition of data to clusters. If not specified, no clusters are

        Intialized:
        -- n_grid: number of hyperparameter grid bins. Default is 30.
        -- distargs: some data types require additional information, for example
        multinomial data requires the number of multinomial categories. See the
        documentation for each type for details
        """
        hypers_grids = cc_datatype_class.construct_hyper_grids(X, n_grid)
        hypers = cc_datatype_class.init_hypers(hypers_grids, X)

        N = len(X)

        self.index = index
        self.N = N
        self.model = cc_datatype_class
        self.X = X
        self.cctype = cc_datatype_class.cctype
        self.hypers_grids = hypers_grids
        self.hypers = hypers
        self.distargs = distargs
        self.mode = 'collapsed'

        if Z is None:
            self.clusters = []
            self.pX = 0
        else:
            self.clusters = []
            K = max(Z)+1
            for k in range(K):
                self.clusters.append(cc_datatype_class(distargs=distargs))
                self.clusters[k].set_hypers(hypers)

            for i in range(len(X)):
                k = Z[i]
                self.clusters[k].insert_element(X[i])

    def predictive_logp(self, n, k):
        """Returns the predictive logp of X[n] in clusters[k]."""
        x = self.X[n]
        lp = self.clusters[k].predictive_logp(x)
        return lp

    def singleton_predictive_logp(self,n):
        """Returns the predictive log_p of X[n] in its own cluster."""
        x = self.X[n]
        lp = self.clusters[0].singleton_logp(x)
        return lp

    def insert_element(self, n, k):
        """Insert X[n] into clusters[k]."""
        x = self.X[n]
        if isnan(x):
            return
        self.clusters[k].insert_element(x)

    def remove_element(self, n, k):
        """Remove X[n] from clusters[k]."""
        x = self.X[n]
        if isnan(x):
            return
        self.clusters[k].remove_element(x)

    def move_to_cluster(self, n, move_from, move_to):
        """Move X[n] from clusters[move_from] to clusters[move_to]."""
        x = self.X[n]
        self.clusters[move_from].remove_element(x)
        self.clusters[move_to].insert_element(x)

    def destroy_singleton_cluster(self, n, to_destroy, move_to):
        """Move X[n] tp clusters[move_to] and destroy clusters[to_destroy]."""
        x = self.X[n]
        self.clusters[move_to].insert_element(x)
        del self.clusters[to_destroy]

    def create_singleton_cluster(self, n, current):
        """Remove X[n] from clusters[current] and create a new singleton
        cluster.
        """
        x = self.X[n]                                               # get the element
        self.clusters[current].remove_element(x)                    # remove from current cluster
        self.clusters.append(self.model(distargs=self.distargs))    # create new empty cluster
        self.clusters[-1].set_hypers(self.hypers)                   # set hypers of new cluster
        self.clusters[-1].insert_element(x)                         # add element to new cluster

    def marginal_logp(self, k):
        """Returns the marginal log_p of clusters[k]."""
        return self.clusters[k].marginal_logp()

    def full_marginal_logp(self):
        """Returns the marginal log_p over all clusters."""
        lp = 0
        for cluster in self.clusters:
            lp += cluster.marginal_logp()

        return lp

    def update_hypers(self):
        """Updates the hyperparameters."""
        self.hypers = self.clusters[0].update_hypers(self.clusters,
            self.hypers_grids)
        for cluster in self.clusters:
            cluster.set_hypers(self.hypers)

    def set_hypers(self, hypers):
        """Set the hyperparameters."""
        self.hypers = hypers
        for cluster in self.clusters:
            cluster.set_hypers(self.hypers)

    def reassign(self, Z):
        """Reassigns the data to new clusters according to the new
        partitioning, Z.

        Destroys and recreates dims.
        """
        self.clusters = []  # destroy dims
        K = max(Z)+1        # get number of clusters in Z

        # create new empty clusters with appropriate args and hypers
        for k in range(K):
            cluster = self.model(distargs=self.distargs)
            cluster.set_hypers(self.hypers)
            self.clusters.append(cluster)

        # insert data into clusters according to Z
        for i in range(self.N):
            k = Z[i]
            self.clusters[k].insert_element(self.X[i])

    def dump_data(self):
        """Removes all data from the clusters. Cleans suffstats."""
        K = len(self.clusters)
        self.clusters = []
        for k in range(K):
            cluster = self.model(distargs=self.distargs)
            cluster.set_hypers(self.hypers)
            self.clusters.append(cluster)

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


    def plot_dist(self):
        """
        Plots the predictive distribution and histogram of X.
        """
        self.model.plot_dist(self.X, self.clusters, distargs=self.distargs)
