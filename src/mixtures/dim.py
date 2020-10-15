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

    def incorporate(self, rowid, observation, inputs=None):
        if rowid in self.Zr or rowid in self.Zi:
            raise ValueError('rowid already incorporated: %d.' % rowid)
        k, inputs_cluster, valid = self.preprocess(observation, None, inputs)
        if k not in self.clusters:
            self.clusters[k] = self.aux_model
            self.aux_model = self.create_aux_model()
        if valid:
            self.clusters[k].incorporate(rowid, observation, inputs_cluster)
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

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        k, inputs2, valid = self.preprocess(targets, constraints, inputs)
        cluster = self.clusters.get(k, self.aux_model)
        # XXX Find out why returning 0 if the query is not valid
        return cluster.logpdf(rowid, targets, constraints, inputs2) \
            if valid else 0

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        k, inputs2, valid = self.preprocess(targets, constraints, inputs)
        cluster = self.clusters.get(k, self.aux_model)
        assert valid
        return cluster.simulate(rowid, targets, constraints, inputs2, N)

    # --------------------------------------------------------------------------
    # Inferece

    def transition_params(self):
        """Transitions the component parameters of each cluster."""
        if not self.is_collapsed():
            for k in self.clusters:
                self.clusters[k].transition_params()

    def transition_hypers(self):
        """Transitions the hyperparameters of each cluster."""
        hypers = list(self.hypers.keys())
        self.rng.shuffle(hypers)
        # For each hyper.
        for hyper in hypers:
            logps = []
            # For each grid point.
            for grid_value in self.hyper_grids[hyper]:
                # Compute the probability of the grid point.
                self.hypers[hyper] = grid_value
                logp_k = 0
                for k in self.clusters:
                    self.clusters[k].set_hypers(self.hypers)
                    logp_k += self.clusters[k].logpdf_score()
                logps.append(logp_k)
            # Sample a new hyperparameter from the grid.
            index = gu.log_pflip(logps, rng=self.rng)
            self.hypers[hyper] = self.hyper_grids[hyper][index]
        # Set the hyperparameters in each cluster.
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
        for model in list(self.clusters.values()):
            model.set_hypers(hypers)

    def get_suffstats(self):
        if len(self.clusters) == 0:
            return {'0' : self.aux_model.get_suffstats()}
        stats = [(str(k), self.clusters[k].get_suffstats()) for k in self.clusters]
        stats_aux = [(str(max(self.clusters) + 1), self.aux_model.get_suffstats())]
        return dict(stats + stats_aux)

    # --------------------------------------------------------------------------
    # Plotter

    def plot_dist(self, X, Y=None, ax=None):
        """Plots predictive distribution and a histogram of data X."""
        from cgpm.utils import plots as pu
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

    def preprocess(self, targets, constraints, inputs):
        inputs2 = inputs.copy()
        try:
            k = inputs2.pop(self.inputs[0])
        except KeyError:
            raise ValueError('Dim needs inputs %d.' % (self.inputs[0],))
        valid_targets = True
        valid_constraints = True
        valid_inputs = True
        if isinstance(targets, dict):
            valid_targets = not math.isnan(targets[self.index])
        if constraints:
            valid_constraints = not any(np.isnan(list(constraints.values())))
        if inputs:
            valid_inputs = not any(np.isnan(list(inputs2.values())))
        assert valid_constraints
        return k, inputs2, valid_targets and valid_inputs
