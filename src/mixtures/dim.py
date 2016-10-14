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

from cgpm.cgpm import CGpm
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import plots as pu


class Dim(CGpm):
    """CGpm representing a homogeneous mixture of univariate CGpm.

    There is no prior over the cluster assignment k of each member. Formally,
    the cluster assignment is a required input variable to the CGpm.
    """

    def __init__(self, outputs, inputs, cctype=None, hypers=None,
            params=None, distargs=None, rng=None):
        """Dim constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data and optional row partition.

        Parameters
        ----------
         cctype : str
             DistributionGpm name see `cgpm.utils.config`.
        outputs : list<int>
            A singleton list containing the identifier of the output variable.
        inputs : list<int>
            A list of at least length 1. The first item is the index of the
            variable corresponding to the required cluster identity. The
            remaining items are input variables to the internal cgpms.
        cctypes : str, optional
            Data type of output variable, defaults to normal.
        hypers : dict, optional
            Shared hypers of internal cgpms.
        params : dict, optional
            Currently disabled.
        distargs : dict, optional.
            Distargs appropriate for the cctype.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        # -- Seed --------------------------------------------------------------
        self.rng = gu.gen_rng() if rng is None else rng

        # -- Outputs -----------------------------------------------------------
        if len(outputs) != 1:
            raise ValueError('Dim requires exactly 1 output.')
        self.outputs = list(outputs)

        # -- Inputs ------------------------------------------------------------
        if len(inputs) < 1:
            raise ValueError('Dim requires at least 1 input.')
        self.inputs = list(inputs)

        # -- Identifier --------------------------------------------------------
        self.index = self.outputs[0]

        # -- DistributionCGpms -------------------------------------------------
        self.model = cu.cctype_class(cctype)
        self.cctype = self.model.name()
        self.distargs = dict(distargs) if distargs is not None else {}

        # -- Hyperparameters ---------------------------------------------------
        self.hyper_grids = {}
        self.hypers = dict(hypers) if hypers is not None else {}

        # -- Clusters and Assignments ------------------------------------------
        self.clusters = {}  # Mapping of cluster k to the object.
        self.Zr = {}        # Mapping of non-nan rowids to cluster k.
        self.Zi = {}        # Mapping of nan rowids to cluster k.

        # -- Auxiliary Singleton ---- ------------------------------------------
        self.aux_model = self.create_aux_model()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate(self, rowid, query, evidence=None):
        if rowid in self.Zr or rowid in self.Zi:
            raise ValueError('rowid already incorporated: %d.' % rowid)
        k, evidence, valid = self.preprocess(query, evidence)
        if k not in self.clusters:
            self.clusters[k] = self.aux_model
            self.aux_model = self.create_aux_model()
        if valid:
            self.clusters[k].incorporate(rowid, query, evidence)
            self.Zr[rowid] = k
        else:
            self.Zi[rowid] = k

    def unincorporate(self, rowid):
        if rowid in self.Zi:
            del self.Zi[rowid]
        elif rowid in self.Zr:
            cluster = self.clusters[self.Zr[rowid]]
            cluster.unincorporate(rowid)
            del self.Zr[rowid]
        else:
            raise ValueError('rowid not incorporated: %d.' % rowid)

    # --------------------------------------------------------------------------
    # logpdf score

    def logpdf_score(self):
        return sum(self.clusters[k].logpdf_score() for k in self.clusters)

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence=None):
        k, evidence, valid = self.preprocess(query, evidence)
        cluster = self.clusters.get(k, self.aux_model)
        return cluster.logpdf(rowid, query, evidence) if valid else 0

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, query, evidence=None, N=None):
        k, evidence, valid = self.preprocess(query, evidence)
        if not valid:
            raise ValueError('Bad simulate args: %s, %s.') % (query, evidence)
        cluster = self.clusters.get(k, self.aux_model)
        return cluster.simulate(rowid, query, evidence, N=N)

    # --------------------------------------------------------------------------
    # Inferece

    def transition_params(self):
        """Transitions the component parameters of each cluster."""
        if not self.is_collapsed():
            for k in self.clusters:
                self.clusters[k].transition_params()

    def transition_hypers(self):
        """Transitions the hyperparameters of each cluster."""
        def transition_hyper(target):
            def compute_cluster_proposal(k, g):
                self.hypers[target] = g
                self.clusters[k].set_hypers(self.hypers)
                return self.clusters[k].logpdf_score()
            def compute_proposal(g):
                return sum(compute_cluster_proposal(k,g) for k in self.clusters)
            logps = [compute_proposal(g) for g in self.hyper_grids[target]]
            index = gu.log_pflip(logps, rng=self.rng)
            self.hypers[target] = self.hyper_grids[target][index]
        # Transition each of the hyperparameters.
        targets = self.hypers.keys()
        self.rng.shuffle(targets)
        for target in targets:
            transition_hyper(target)
        for k in self.clusters:
            self.clusters[k].set_hypers(self.hypers)
        self.aux_model = self.create_aux_model()

    def transition_hyper_grids(self, X, n_grid=30):
        """Transitions hyperparameter grids using empirical Bayes."""
        self.hyper_grids = self.model.construct_hyper_grids(
            [x for x in X if not math.isnan(x)], n_grid=n_grid)
        # Only transition the hypers if previously uninstantiated.
        if not self.hypers:
            for h in self.hyper_grids:
                self.hypers[h] = self.rng.choice(self.hyper_grids[h])
        self.aux_model = self.create_aux_model()

    # --------------------------------------------------------------------------
    # Attributes from self.model

    def get_distargs(self):
        return self.aux_model.get_distargs()

    def is_collapsed(self):
        return self.model.is_collapsed()

    def is_conditional(self):
        return self.model.is_conditional()

    def is_continuous(self):
        return self.model.is_continuous()

    def is_numeric(self):
        return self.model.is_numeric()

    def name(self):
        return self.aux_model.name()

    def set_hypers(self, hypers):
        self.hypers = hypers
        for model in self.clusters.values():
            model.set_hypers(hypers)

    # --------------------------------------------------------------------------
    # Plotter

    def plot_dist(self, X, Y=None, ax=None):
        """Plots predictive distribution and a histogram of data X."""
        plotter = pu.plot_dist_continuous if self.model.is_continuous() else \
            pu.plot_dist_discrete
        return plotter(
            [x for x in X if not math.isnan(x)], self.index, self.clusters,
            ax=ax, Y=Y, hist=False)

    # --------------------------------------------------------------------------
    # Internal

    def create_aux_model(self):
        return self.model(
            outputs=[self.index], inputs=self.inputs[1:], hypers=self.hypers,
            distargs=self.distargs, rng=self.rng)

    def preprocess(self, query, evidence):
        evidence = evidence.copy()
        try:
            k = evidence.pop(self.inputs[0])
        except KeyError:
            raise ValueError('Dim needs inputs[0] in evidence: %s' % evidence)
        valid_x = True
        valid_y = True
        if isinstance(query, dict):
            valid_x = not math.isnan(query[self.index])
        if evidence:
            valid_y = not any(np.isnan(evidence.values()))
        return k, evidence, valid_x and valid_y
