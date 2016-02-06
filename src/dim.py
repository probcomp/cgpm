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

import numpy as np
from math import isnan

import gpmcc.utils.config as cu
import gpmcc.utils.general as gu
import gpmcc.utils.plots as pu

class Dim(object):
    """Dim, holds suff stats, DistributionGpm type, clusters, and shared
    hyperparameters and grids."""

    def __init__(self, X, cctype, index, distargs=None, Zr=None, n_grid=30,
            hypers=None):
        """Dim constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data and optional row partition.

        Parameters
        ----------
        X : np.array
            Array of data. Must be compatible with `cctype`. Missing entries
            must be np.nan. The dataset X is summarized by the sufficient
            statistics only and is not stored.
        cctype : str
            DistributionGpm name see `gpmcc.utils.config`.
        index : int
            Unique identifier for this dim.
        distargs : dict, optional.
            Distargs appropriate for the cctype. For details on
            distargs see the documentation for each DistributionGpm.
        Zr : list<int>, optional
            Partition of data X into clusters, where Zr[i] is the cluster
            index of row X[i]. If None, intialized from CRP(1). The partition
            is only for initialization and is not stored.
        n_grid : int, optional
            Number of bins in the hyperparameter grid.
        """
        # Identifier.
        self.index = index

        # Model type.
        self.model = cu.cctype_class(cctype)
        self.cctype = self.model.name()
        self.distargs = distargs if distargs is not None else {}

        # Hyperparams.
        self.transition_hyper_grids(X, n_grid)
        self.hypers = hypers
        if hypers is None:
            self.hypers = dict()
            for h in self.hyper_grids:
                self.hypers[h] = np.random.choice(self.hyper_grids[h])

        # Row partition.
        if Zr is None:
            Zr = gu.simulate_crp(len(X), 1)
        self.reassign(X, Zr)

        # Auxiliary singleton model.
        self.aux_model = self.model(distargs=self.distargs, **self.hypers)

    # --------------------------------------------------------------------------
    # Observe

    def incorporate(self, x, k):
        """Record an observation x in clusters[k].
        If k < len(self.clusters) then x will be incorporated to cluster k.
        If k == len(self.clusters) a new cluster will be created.
        If k > len(self.clusters) an error will be thrown.
        """
        assert k <= len(self.clusters)
        if k == len(self.clusters):
            self.clusters.append(self.aux_model)
            self.aux_model = self.model(distargs=self.distargs,
                **self.hypers)
        if not isnan(x):
            self.clusters[k].incorporate(x)

    def unincorporate(self, x, k):
        """Remove observation x from clusters[k]. Bad things will happen if x
        was not incorporated into cluster k before calling this method.
        """
        assert k < len(self.clusters)
        if not isnan(x):
            self.clusters[k].unincorporate(x)

    # Bulk operations for effeciency.

    def destroy_cluster(self, k):
        """Destroy cluster k, and all its incorporated data (if any)."""
        assert k < len(self.clusters)
        del self.clusters[k]

    def reassign(self, X, Zr):
        """Reassigns data X to new clusters according to partitioning Zr.
        Destroys and recreates all clusters. Uncollapsed parameters are
        transitioned but hyperparameters are not transitioned. The partition
        is only for reassigning, and not stored internally.
        """
        assert len(X) == len(Zr)
        self.clusters = []
        K = max(Zr) + 1

        # Create clusters.
        for k in xrange(K):
            cluster = self.model(distargs=self.distargs, **self.hypers)
            self.clusters.append(cluster)

        # Populate clusters.
        for x, k in zip(X, Zr):
            if not isnan(x):
                self.clusters[k].incorporate(x)

        # Transition uncollapsed params if necessary.
        if not self.is_collapsed():
            for cluster in self.clusters:
                cluster.transition_params()

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, x, k):
        """Returns the predictive logp of x in clusters[k]. If x has been
        assigned to clusters[k], then use the unincorporate/incorporate
        interface to compute the true predictive logp."""
        if k == len(self.clusters):
            # Good for inference quality, always uses latest hypers.
            self.aux_model = self.model(distargs=self.distargs,
                **self.hypers)
            cluster = self.aux_model
        else:
            cluster = self.clusters[k]
        return cluster.logpdf(x) if not isnan(x) else 0

    def logpdf_marginal(self, k=None):
        """If k is not None, returns the marginal log_p of clusters[k].
        Otherwise returns the sum of marginal log_p over all clusters."""
        if k is not None:
            return self.clusters[k].logpdf_marginal()
        return [cluster.logpdf_marginal() for cluster in self.clusters]

    # --------------------------------------------------------------------------
    # Simulate
    def simulate(self, k):
        """If k is not None, returns the marginal log_p of clusters[k].
        Otherwise returns the sum of marginal log_p over all clusters."""
        if k == len(self.clusters):
            # Good for inference quality, always uses latest hypers.
            self.aux_model = self.model(distargs=self.distargs,
                **self.hypers)
            cluster = self.aux_model
        else:
            cluster = self.clusters[k]
        return cluster.simulate()

    # --------------------------------------------------------------------------
    # Inferece

    def transition_params(self):
        """Updates the component parameters of each cluster."""
        if not self.is_collapsed():
            for cluster in self.clusters:
                cluster.transition_params()

    def transition_hypers(self):
        """Updates the hyperparameters of each cluster."""
        for cluster in self.clusters:
            cluster.set_hypers(self.hypers)
        targets = self.hypers.keys()
        np.random.shuffle(targets)
        for target in targets:
            logps = self._calc_hyper_proposal_logps(target)
            proposal = gu.log_pflip(logps)
            self.hypers[target] = self.hyper_grids[target][proposal]

    def transition_hyper_grids(self, X, n_grid):
        """Resample the hyperparameter grids using empirical Bayes."""
        self.hyper_grids = self.model.construct_hyper_grids(
            X[~np.isnan(X)], n_grid)

    # --------------------------------------------------------------------------
    # Helpers

    def get_suffstats(self):
        return [cluster.get_suffstats() for cluster in self.clusters]

    def is_collapsed(self):
        return self.model.is_collapsed()

    def plot_dist(self, X, Y=None, ax=None):
        """Plots the predictive distribution and histogram of X."""
        plotter = pu.plot_dist_continuous if self.model.is_continuous() else \
            pu.plot_dist_discrete
        return plotter(X[~np.isnan(X)], self.clusters, ax=ax, Y=Y, hist=False)

    # --------------------------------------------------------------------------
    # Internal

    def _calc_hyper_proposal_logps(self, target):
        """Computes the marginal likelihood (over all clusters) for each
        hyperparameter value in self.hyper_grids[target].
        p(h|X) \prop p(h)p(X|h)
        """
        logps = []
        hypers = self.hypers.copy()
        for g in self.hyper_grids[target]:
            hypers[target] = g
            logp = 0
            for cluster in self.clusters:
                cluster.set_hypers(hypers)
                logp += cluster.logpdf_marginal()
                cluster.set_hypers(self.hypers)
            logps.append(logp)
        return logps
