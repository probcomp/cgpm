# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np

import gpmcc.utils.config as cu
import gpmcc.utils.general as gu
import gpmcc.utils.plots as pu


class Dim(object):
    """Dim holds sufficient statistics, DistributionGpm type, clusters, and
    shared hyperparameters and grids. Technically not GPM, but easily becomes
    one by placing creating a View with a single Dim."""

    def __init__(self, outputs, inputs=None, cctype=None, hypers=None,
            params=None, distargs=None, rng=None):
        """Dim constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data and optional row partition.

        Parameters
        ----------
        cctype : str
            DistributionGpm name see `gpmcc.utils.config`.
        index : int
            Unique identifier for this dim.
        distargs : dict, optional.
            Distargs appropriate for the cctype. For details on
            distargs see the documentation for each DistributionGpm.
        """
        self.rng = gu.gen_rng() if rng is None else rng

        # Identifier.
        if len(outputs) != 1:
            raise ValueError('Dim requires exactly 1 output.')
        self.outputs = outputs
        self.inputs = inputs if inputs else []
        # XXX ENCAPSULATE ME!
        self.index = self.outputs[0]

        # Model type.
        self.model = cu.cctype_class(cctype)
        self.cctype = self.model.name()
        self.distargs = distargs if distargs is not None else {}

        # Hyperparams.
        self.hyper_grids = {}
        self.hypers = hypers if hypers is not None else {}

        # Clusters.
        self.clusters = []
        self.clusters_inverse = {}
        self.ignored = set([])

        # Auxiliary singleton model.
        self.aux_model = self.create_aux_model()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate(self, rowid, query, evidence):
        """Record an observation.

        If k < len(self.clusters) then x will be incorporated to cluster k.
        If k == len(self.clusters) then a new cluster will be created.
        If k > len(self.clusters) then an error will be thrown.
        """
        k, evidence, valid = self.preprocess(query, evidence)
        assert k <= len(self.clusters)
        if k == len(self.clusters):
            self.clusters.append(self.aux_model)
            self.aux_model = self.create_aux_model()
        if valid:
            self.clusters[k].incorporate(rowid, query, evidence)
            self.clusters_inverse[rowid] = self.clusters[k]
        else:
            self.ignored.add(rowid)

    def unincorporate(self, rowid):
        """Remove observation rowid."""
        if rowid in self.ignored:
            self.ignored.remove(rowid)
        else:
            cluster = self.clusters_inverse[rowid]
            cluster.unincorporate(rowid)
            del self.clusters_inverse[rowid]

    # --------------------------------------------------------------------------
    # Github issue #65.

    def logpdf_score(self, k=None):
        """Return log score summed over all clusters."""
        return sum(cluster.logpdf_score() for cluster in self.clusters)

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence):
        """Evaluate the log density of the query given evidence."""
        k, evidence, valid = self.preprocess(query, evidence)
        cluster = self.aux_model if k==len(self.clusters) else self.clusters[k]
        return cluster.logpdf(rowid, query, evidence) if valid else 0

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, query, evidence):
        """Simulate the query given evidence."""
        k, evidence, valid = self.preprocess(query, evidence)
        if not valid:
            raise ValueError('Bad simulate args: %s, %s.') % (query, evidence)
        cluster = self.aux_model if k==len(self.clusters) else self.clusters[k]
        return cluster.simulate(rowid, query, evidence)

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
        self.rng.shuffle(targets)
        for target in targets:
            logps = self._calc_hyper_proposal_logps(target)
            proposal = gu.log_pflip(logps, rng=self.rng)
            self.hypers[target] = self.hyper_grids[target][proposal]

    def transition_hyper_grids(self, X, n_grid=30):
        """Resample the hyperparameter grids using empirical Bayes."""
        self.hyper_grids = self.model.construct_hyper_grids(
            [x for x in X if not math.isnan(x)], n_grid=n_grid)
        # Only transition the hypers if previously uninstantiated.
        if not self.hypers:
            for h in self.hyper_grids:
                self.hypers[h] = self.rng.choice(self.hyper_grids[h])
        self.aux_model = self.create_aux_model()

    # --------------------------------------------------------------------------
    # Helpers

    def is_collapsed(self):
        return self.model.is_collapsed()

    def is_continuous(self):
        return self.model.is_continuous()

    def is_conditional(self):
        return self.model.is_conditional()

    def is_numeric(self):
        return self.model.is_numeric()

    def get_distargs(self):
        return self.aux_model.get_distargs()

    def name(self):
        return self.aux_model.get_name()

    def plot_dist(self, X, Y=None, ax=None):
        """Plots the predictive distribution and histogram of X."""
        plotter = pu.plot_dist_continuous if self.model.is_continuous() else \
            pu.plot_dist_discrete
        return plotter(
            [x for x in X if not math.isnan(x)], self.index, self.clusters,
            ax=ax, Y=Y, hist=False)

    # --------------------------------------------------------------------------
    # Internal

    def create_aux_model(self):
        return self.model(
            outputs=[self.index], inputs=self.inputs, hypers=self.hypers,
            distargs=self.distargs, rng=self.rng)

    def preprocess(self, query, evidence):
        evidence = evidence.copy()
        try:
            k = evidence.pop(-1)
        except KeyError:
            raise ValueError('Dim needs cluster -1 in evidence: %s' % evidence)
        valid_x, valid_y = True, True
        if isinstance(query, dict):
            valid_x = not math.isnan(query[self.index])
        if evidence:
            valid_y = not any(np.isnan(evidence.values()))
        return k, evidence, valid_x and valid_y

    def _calc_hyper_proposal_logps(self, target):
        """Computes the log score for each value in self.hyper_grids[target].

        p(h|X) \\prop p(X,h).
        """
        logps = []
        hypers = self.hypers.copy()
        for g in self.hyper_grids[target]:
            hypers[target] = g
            logp = 0
            for cluster in self.clusters:
                cluster.set_hypers(hypers)
                logp += cluster.logpdf_score()
                cluster.set_hypers(self.hypers)
            logps.append(logp)
        return logps
