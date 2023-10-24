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

from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import zip
from builtins import range
from past.utils import old_div
import pickle as pickle
import copy
import importlib
import itertools
import sys
import time

from collections import OrderedDict
from collections import defaultdict
from math import isnan

import numpy as np

from cgpm.cgpm import CGpm
from cgpm.crosscat import sampling
from cgpm.mixtures.dim import Dim
from cgpm.mixtures.view import View
from cgpm.network.helpers import retrieve_ancestors
from cgpm.network.helpers import retrieve_variable_to_cgpm
from cgpm.network.helpers import retrieve_weakly_connected_components
from cgpm.network.importance import ImportanceNetwork
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import timer as tu
from cgpm.utils import validation as vu


class State(CGpm):
    """CGpm representing Crosscat, built as a composition of smaller CGpms."""

    def __init__(
            self, X, outputs=None, inputs=None, cctypes=None,
            distargs=None, Zv=None, Zrv=None, structure_hypers=None, view_structure_hypers=None,
            hypers=None, Cd=None, Ci=None, Rd=None, Ri=None, diagnostics=None,
            loom_path=None, rng=None):
        """ Construct a State.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, each row is an observation and each column a variable.
        outputs : list<int>, optional
            Unique non-negative ID for each column in X, and used to refer to
            the column for all future queries. Defaults to range(0, X.shape[1])
        inputs : list<int>, optional
            Currently unsupported.
        cctypes : list<str>
            Data type of each column, see `utils.config` for valid cctypes.
        distargs : list<dict>, optional
            See the documentation for each DistributionGpm for its distargs.
        Zv : dict(int:int), optional
            Assignment of output columns to views, where Zv[k] is the
            view assignment for column k. Defaults to sampling from CRP.
        Zrv : dict(int:list<int>), optional
            Assignment of rows to clusters in each view, where Zrv[k] is
            the Zr for View k. If specified, then Zv must also be specified.
            Defaults to sampling from CRP.
        Cd : list(list<int>), optional
            List of marginal dependence constraints for columns. Each element in
            the list is a list of columns which are to be in the same view. Each
            column can only be in one such list i.e. [[1,2,5],[1,5]] is not
            allowed.
        Ci : list(tuple<int>), optional
            List of marginal independence constraints for columns.
            Each element in the list is a 2-tuple of columns that must be
            independent, i.e. [(1,2),(1,3)].
        Rd : dict(int:Cd), optional
            Dictionary of dependence constraints for rows, wrt.
            Each entry is (col: Cd), where col is a column number and Cd is a
            list of dependence constraints for the rows with respect to that
            column (see doc for Cd).
        Ri : dict(int:Cid), optional
            Dictionary of independence constraints for rows, wrt.
            Each entry is (col: Ci), where col is a column number and Ci is a
            list of independence constraints for the rows with respect to that
            column (see doc for Ci).
        iterations : dict(str:int), optional
            Metadata holding the number of iters each kernel has been run.
        loom_path: str, optional
            Path to a loom project compatible with this State.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        # -- Seed --------------------------------------------------------------
        self.rng = gu.gen_rng() if rng is None else rng

        # -- Inputs ------------------------------------------------------------
        if inputs:
            raise ValueError('State does not accept inputs.')
        self.inputs = []

        # -- Dataset and outputs -----------------------------------------------
        X = np.asarray(X)
        if not outputs:
            outputs = list(range(X.shape[1]))
        else:
            assert len(outputs) == X.shape[1]
            assert all(o >= 0 for o in outputs)
        self.set_outputs(outputs)
        self.X = OrderedDict()
        for i, c in enumerate(self.outputs):
            self.X[c] = X[:,i].tolist()

        # -- Column CRP --------------------------------------------------------
        # Retrieve the dependence constraints.
        if Rd is not None:
            raise ValueError('Row dependence constraints not implemented.')
        if Ri is not None:
            raise ValueError('Row independence constraints not implemented.')
        self.Cd = [] if Cd is None else Cd
        self.Ci = [] if Ci is None else Ci
        self.Rd = {}
        self.Ri = {}
        # Prepare the GPM for the column crp.
        self.crp_id = 5**8
        self.crp = Dim(
            outputs=[self.crp_id],
            inputs=[-1],
            cctype='crp',
            hypers=structure_hypers,
            rng=self.rng
        )
        self.crp.transition_hyper_grids([1]*self.n_cols())
        # Simulate a CRP for the column partition.
        if Zv is None:
            # Simulate a constrained CRP.
            if self.Ci or self.Cd:
                # Require outputs are zero-based for now, rather than worry
                # about maintaining a zero-based map.
                if self.outputs != list(range(self.n_cols())):
                    raise ValueError('Use zero-based outputs with constraints.')
                if self.Ci:
                    # Independence constraints are specified; simulate
                    # the non-exchangeable version of the constrained crp.
                    Zv = gu.simulate_crp_constrained(
                        self.n_cols(), 
                        self.crp.hypers['alpha'], 
                        self.crp.hypers['discount'],
                        self.Cd, self.Ci,
                        self.Rd, self.Ri, rng=self.rng)
                else:
                    # Only dependence constraints are specified; simulate
                    # the exchangeable version of the constrained crp.
                    Zv = gu.simulate_crp_constrained_dependent(
                        self.n_cols(), 
                        self.crp.hypers['alpha'], 
                        self.crp.hypers['discount'],
                        self.Cd, self.rng)
                # Incorporate hte the data.
                for c, z in zip(self.outputs, Zv):
                    self.crp.incorporate(c, {self.crp_id: z}, {-1:0})
            # Otherwise simulate an unconstrained CRP.
            else:
                for c in self.outputs:
                    z = self.crp.simulate(c, [self.crp_id], None, {-1:0})
                    self.crp.incorporate(c, z, {-1:0})
        # Load the provided Zv without simulation.
        else:
            for c, z in Zv.items():
                self.crp.incorporate(c, {self.crp_id: z}, {-1:0})

        assert len(self.Zv()) == len(self.outputs)

        # -- View data ---------------------------------------------------------
        cctypes = cctypes or [None] * len(self.outputs)
        distargs = distargs or [None] * len(self.outputs)
        hypers = hypers or [None] * len(self.outputs)
        view_structure_hypers = view_structure_hypers or {}

        # If the user specifies Zrv, then the keys of Zrv must match the views
        # which are values in Zv.
        if Zrv is None:
            Zrv = {}
        else:
            assert set(Zrv.keys()) == set(self.Zv().values())

        # -- Views -------------------------------------------------------------
        self.views = OrderedDict()
        self.crp_id_view = 10**7
        for v in set(self.Zv().values()):
            v_outputs = [o for o in self.outputs if self.Zv(o) == v]
            v_cctypes = [cctypes[self.outputs.index(c)] for c in v_outputs]
            v_distargs = [distargs[self.outputs.index(c)] for c in v_outputs]
            v_hypers = [hypers[self.outputs.index(c)] for c in v_outputs]
            view = View(
                self.X,
                outputs=[self.crp_id_view+v] + v_outputs,
                inputs=None,
                Zr=Zrv.get(v, None),
                structure_hypers=view_structure_hypers.get(v, None),
                cctypes=v_cctypes,
                distargs=v_distargs,
                hypers=v_hypers,
                rng=self.rng
            )
            self.views[v] = view

        # -- Foreign CGpms -----------------------------------------------------
        self.token_generator = itertools.count(start=57481)
        self.hooked_cgpms = dict()

        # -- Diagnostic Checkpoints---------------------------------------------
        if diagnostics is None:
            self.diagnostics = defaultdict(list)
            self.diagnostics['iterations'] = dict()
        else:
            self.diagnostics = defaultdict(list, diagnostics)

        # -- Loom project ------------------------------------------------------
        self._loom_path = loom_path

        # -- Validate ----------------------------------------------------------
        self._check_partitions()

        # -- Composite ---------------------------------------------------------
        # Does the state have any conditional GPMs? Conditional GPMs come from
        # - a hooked cgpm;
        # - a conditional dim.
        self._composite = False

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(
            self, T, outputs, inputs=None, cctype=None, distargs=None, v=None):
        """Incorporate a new Dim into this State with data T.

        Parameters
        ----------
        T : list
            Data with length self.n_rows().
        outputs : list[int]
            Identity of the variable modeled by this dim, must be non-negative
            and cannot collide with State.outputs. Only univariate outputs
            currently supported, so the list be a singleton.
        cctype, distargs:
            refer to State.__init__
        v : int, optional
            Index of the view to assign the data. If 0 <= v < len(state.views)
            then insert into an existing View. If v = len(state.views) then
            singleton view will be created with a partition from the CRP prior.
            If unspecified, will be sampled.
        """
        if len(T) != self.n_rows():
            raise ValueError(
                '%d rows are required, received: %d.'
                % (self.n_rows(), len(T)))
        if len(outputs) != 1:
            raise ValueError(
                'Cannot incorporate multivariate outputs: %s.'
                % outputs)
        if outputs[0] in self.outputs:
            raise ValueError(
                'Specified outputs already exist: %s, %s.'
                % (outputs, self.outputs))
        if inputs:
            raise ValueError(
                'Cannot incorporate dim with inputs: %s.'
                % inputs)
        # Append new output to outputs.
        col = outputs[0]
        self.X[col] = T
        self.set_outputs(self.outputs + [col])
        # If v unspecified then transition the col.
        transition = [col] if v is None else []
        # Determine correct view.
        v_add = 0 if v is None else v
        if v_add in self.views:
            view = self.views[v_add]
        else:
            view = View(self.X, outputs=[self.crp_id_view + v_add], rng=self.rng)
            self._append_view(view, v_add)
        # Create the dimension.
        # XXX Does not handle conditional models; consider moving to view?
        D = Dim(
            outputs=outputs,
            inputs=[view.outputs[0]],
            cctype=cctype,
            distargs=distargs,
            rng=self.rng
        )
        D.transition_hyper_grids(self.X[col])
        view.incorporate_dim(D)
        self.crp.incorporate(col, {self.crp_id: v_add}, {-1:0})
        # Transition.
        self.transition_dims(cols=transition)
        self.transition_dim_hypers(cols=[col])
        # Update composite flag.
        self._update_is_composite()
        # Validate.
        self._check_partitions()

    def unincorporate_dim(self, col):
        """Unincorporate the Dim whose output[0] is col."""
        if self.n_cols() == 1:
            raise ValueError('State has only one dim, cannot unincorporate.')
        if col not in self.outputs:
            raise ValueError('col does not exist: %s, %s.')
        # Find the dim and its view.
        d_del = self.dim_for(col)
        v_del = self.Zv(col)
        delete = self.Nv(v_del) == 1
        self.views[v_del].unincorporate_dim(d_del)
        self.crp.unincorporate(col)
        # Clear a singleton.
        if delete:
            self._delete_view(v_del)
        # Clear data for col and remove it from outputs.
        del self.X[col]
        self.set_outputs([i for i in self.outputs if i != col])
        # Update composite flag.
        self._update_is_composite()
        # Validate.
        self._check_partitions()

    def incorporate(self, rowid, observation, inputs=None):
        # XXX Only allow new rows for now.
        if rowid != self.n_rows():
            raise ValueError('Only contiguous rowids supported: %d' % (rowid,))
        if inputs:
            raise ValueError('Cannot incorporate with inputs: %s' % inputs)
        valid_clusters = set([self.views[v].outputs[0] for v in self.views])
        query_clusters = [q for q in observation if q in valid_clusters]
        query_outputs = [q for q in observation if q not in query_clusters]
        if not all(q in self.outputs for q in query_outputs):
            raise ValueError('Invalid observation: %s' % observation)
        if any(isnan(v) for v in list(observation.values())):
            raise ValueError('Cannot incorporate nan: %s.' % observation)
        # Append the observation to dataset.
        for c in self.outputs:
            self.X[c].append(observation.get(c, float('nan')))
        # Pick a fresh rowid.
        if self.hypothetical(rowid):
            rowid = self.n_rows()-1
        # Tell the views.
        for v in self.views:
            query_v = {d: self.X[d][rowid] for d in self.views[v].dims}
            crp_v = self.views[v].outputs[0]
            cluster_v = {crp_v: observation[crp_v]} if crp_v in observation\
                else {}
            self.views[v].incorporate(rowid, gu.merged(cluster_v, query_v))
        # Validate.
        self._check_partitions()

    def unincorporate(self, rowid):
        # XXX WHATTA HACK. Only permit unincorporate the last rowid, which means
        # we can pop the last entry of each list in self.X without affecting any
        # existing rowids.
        if rowid != self.n_rows() - 1:
            raise ValueError('Only last rowid may be unincorporated.')
        if self.n_rows() == 1:
            raise ValueError('Cannot unincorporate last rowid.')
        # Remove the observation from the dataset.
        for c in self.outputs:
            self.X[c].pop()
        # Tell the views.
        for v in self.views:
            self.views[v].unincorporate(rowid)
        # Validate.
        self._check_partitions()

    # XXX Major hack to force values of NaN cells in incorporated rowids.
    def force_cell(self, rowid, observation):
        if (rowid is None):
            raise ValueError('Force observation requires existing rowid.')
        if not 0 <= rowid < self.n_rows():
            raise ValueError('Force observation requires existing rowid.')
        if not all(np.isnan(self.X[c][rowid]) for c in observation):
            raise ValueError('Force observations requires NaN cells.')
        for col, value in observation.items():
            self.X[col][rowid] = value
        queries = vu.partition_list(
            {c: self.Zv(c) for c in observation}, observation)
        for view_id, view_variables in queries.items():
            observation_v = {c: observation[c] for c in view_variables}
            self.views[view_id].force_cell(rowid, observation_v)

    # --------------------------------------------------------------------------
    # Schema updates.

    def update_cctype(self, col, cctype, distargs=None):
        """Update the distribution type of self.dims[col] to cctype.

        Parameters
        ----------
        col : int
            Index of column to update.
        cctype, distargs:
            refer to State.__init__
        """
        assert col in self.outputs
        self.view_for(col).update_cctype(col, cctype, distargs=distargs)
        self.transition_dim_grids(cols=[col])
        self.transition_dim_params(cols=[col])
        self.transition_dim_hypers(cols=[col])
        # Update composite flag.
        self._update_is_composite()
        # Validate.
        self._check_partitions()

    # --------------------------------------------------------------------------
    # Compositions.

    def compose_cgpm(self, cgpm):
        """Compose a CGPM with this object. Returns `token` to be used in the
        call to decompose_cgpm.

        Parameters
        ----------
        cgpm : cgpm.cgpm.CGpm object
            The `CGpm` object to compose.

        Returns
        -------
        token : int
            A unique token representing the composed cgpm, to be used
            by `State.decompose_cgpm`.
        """
        token = next(self.token_generator)
        self.hooked_cgpms[token] = cgpm
        try:
            self.build_network()
        except ValueError as e:
            del self.hooked_cgpms[token]
            raise e
        self._update_is_composite()
        return token

    def decompose_cgpm(self, token):
        """Decompose a previously composed CGPM with identifier `token`.

        Parameters
        ----------
        token : int
            The unique token representing the composed cgpm, returned from
            `State.compose_cgpm`.
        """
        del self.hooked_cgpms[token]
        self._update_is_composite()
        self.build_network()

    def _update_is_composite(self):
        """Update state._composite attribute."""
        hooked = len(self.hooked_cgpms) > 0
        conditional = any(d.is_conditional() for d in self.dims())
        self._composite = hooked or conditional

    def is_composite(self):
        return self._composite

    # --------------------------------------------------------------------------
    # logscore.

    def logpdf_score_crp(self):
        if self.Ci:
            raise ValueError('Cannot compute crp score with independences.')
        # Compute vanilla CRP probability.
        if not self.Cd:
            return self.crp.logpdf_score()
        # Compute constrained CRP probability.
        Zv, alpha, discount = self.Zv(), self.crp.hypers['alpha'], self.crp.hypers['discount']
        return gu.logp_crp_constrained_dependent(Zv, alpha, discount, self.Cd)

    def logpdf_likelihood(self):
        logp_views = sum(v.logpdf_likelihood() for v in self.views.values())
        return logp_views

    def logpdf_score(self):
        """Compute joint density of all latents and the incorporated data.

        Returns
        -------
        logpdf_score : float
            The log score is P(X,Z) = P(X|Z)P(Z) where X is the observed data
            and Z is the entirety of the latent state in the CGPM.
        """
        logp_crp = self.logpdf_score_crp()
        logp_views = sum(v.logpdf_score() for v in self.views.values())
        return logp_crp + logp_views

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, targets, constraints=None, inputs=None,
            accuracy=None):
        assert isinstance(targets, dict)
        assert constraints is None or isinstance(constraints, dict)
        self._validate_cgpm_query(rowid, targets, constraints)
        if not self._composite:
            assert not inputs
            return sampling.state_logpdf(self, rowid, targets, constraints)
        constraints = self._populate_constraints(rowid, targets, constraints)
        network = self.build_network(accuracy=accuracy)
        return network.logpdf(rowid, targets, constraints, inputs)

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, targets, constraints=None, inputs=None,
            N=None, accuracy=None):
        assert isinstance(targets, (list, tuple))
        assert inputs is None or isinstance(inputs, dict)
        self._validate_cgpm_query(rowid, targets, constraints)
        if not self._composite:
            assert not inputs
            return sampling.state_simulate(self, rowid, targets, constraints, N)
        constraints = self._populate_constraints(rowid, targets, constraints)
        network = self.build_network(accuracy=accuracy)
        return network.simulate(rowid, targets, constraints, inputs, N)

    # --------------------------------------------------------------------------
    # simulate/logpdf helpers

    def build_network(self, accuracy=None):
        if accuracy is None: accuracy=1
        return ImportanceNetwork(self.build_cgpms(), accuracy, rng=self.rng)

    def build_cgpms(self):
        return [self.views[v] for v in self.views] + list(self.hooked_cgpms.values())

    def _populate_constraints(self, rowid, targets, constraints):
        """Loads constraints from the dataset."""
        constraints = constraints or dict()
        # If the rowid is hypothetical, just return.
        if self.hypothetical(rowid):
            return constraints
        # Retrieve all values for this rowid not in targets or constraints.
        data = {
            c: self.X[c][rowid]
            for c in self.outputs[1:]
            if not any([
                c in targets,
                c in constraints,
                isnan(self.X[c][rowid]),
            ])
        }
        return gu.merged(constraints, data)

    def _validate_cgpm_query(self, rowid, targets, constraints):
        # Is the rowid fresh?
        fresh = self.hypothetical(rowid)
        # Is the query simulate or logpdf?
        simulate = isinstance(targets, (list, tuple))
        # Disallow duplicated target cols.
        if simulate and len(set(targets)) != len(targets):
            raise ValueError('Columns in targets must be unique.')
        # Disallow query constraining observed cells.
        # XXX Only disallow logpdf constraints; simulate is permitted for
        # INFER EXPLICIT PREDICT through BQL to work. Refer to
        # https://github.com/probcomp/cgpm/issues/116
        if not fresh \
                and not simulate \
                and any(not np.isnan(self.X[q][rowid]) for q in targets):
            raise ValueError('Cannot constrain observed cell.')
        # Check if the constraints is valid.
        if constraints:
            # Disallow overlap between targets and constraints.
            if len(set.intersection(set(targets), set(constraints))) > 0:
                raise ValueError('Targets and constraints must be disjoint.')
            # Disallow constraints specifying with observed cells.
            def good_constraint(rowid, e):
                return \
                    e not in self.outputs \
                    or np.isnan(self.X[e][rowid]) \
                    or np.allclose(self.X[e][rowid], constraints[e])
            if not fresh \
                    and any(not good_constraint(rowid, e) for e in constraints):
                raise ValueError('Cannot use observed cell in constraints.')

    # --------------------------------------------------------------------------
    # Bulk operations for multiprocessing performance.

    def simulate_bulk(self, rowids, targets_list, constraints_list=None,
            inputs_list=None, Ns=None):
        """Evaluate multiple queries at once, used by Engine."""
        if constraints_list is None:
            constraints_list = [{} for i in range(len(rowids))]
        if inputs_list is None:
            inputs_list = [{} for i in range(len(rowids))]
        if Ns is None:
            Ns = [1 for i in range(len(rowids))]
        assert len(rowids) == len(targets_list)
        assert len(rowids) == len(constraints_list)
        assert len(rowids) == len(inputs_list)
        assert len(rowids) == len(Ns)
        return [
            self.simulate(r, t, c, i, n)
            for (r, t, c, i, n) in zip(
                rowids,
                targets_list,
                constraints_list,
                inputs_list,
                Ns
            )
        ]

    def logpdf_bulk(self, rowids, targets_list, constraints_list=None,
            inputs_list=None):
        """Evaluate multiple queries at once, used by Engine."""
        if constraints_list is None:
            constraints_list = [{} for i in range(len(rowids))]
        if inputs_list is None:
            inputs_list = [{} for i in range(len(rowids))]
        assert len(rowids) == len(targets_list)
        assert len(rowids) == len(constraints_list)
        assert len(rowids) == len(inputs_list)
        return [
            self.logpdf(r, t, c, i)
            for (r, t, c, i) in zip(
                rowids,
                targets_list,
                constraints_list,
                inputs_list
            )
        ]

    def incorporate_bulk(self, rowids, observations, inputs=None):
        """Incorporate multiple observations at once, used by Engine."""
        for rowid, observation in zip(rowids, observations):
            self.incorporate(rowid, observation, inputs)

    def force_cell_bulk(self, rowids, queries):
        """Force multiple cell values at once, used by Engine."""
        for rowid, query in zip(rowids, queries):
            self.force_cell(rowid, query)

    # --------------------------------------------------------------------------
    # Dependence probability.

    def dependence_probability(self, col0, col1):
        # Use the CrossCat view partition for state variables.
        if self.has_output(col0) and self.has_output(col1):
            return float(self.Zv(col0) == self.Zv(col1))
        Zv = {i: self.Zv(i) for i in self.outputs}
        cgpms = self.build_cgpms()
        return State._dependence_probability_composite(cgpms, Zv, col0, col1)

    def dependence_probability_pairwise(self, colnos=None):
        if colnos is None:
            colnos = self.outputs
        D = np.eye(len(colnos))
        reindex = {c: k for k, c in enumerate(colnos)}
        for i,j in itertools.combinations(colnos, 2):
            d = self.dependence_probability(i, j)
            D[reindex[i], reindex[j]] = D[reindex[j], reindex[i]] = d
        return D

    @staticmethod
    def _dependence_probability_composite(cgpms, Zv, col0, col1):
        # XXX Conservatively assume all outputs of a particular are dependent.
        if any(col0 in c.outputs and col1 in c.outputs for c in cgpms):
            return 1.
        # Use the BayesBall algorithm on the cgpm network.
        ancestors0 = retrieve_ancestors(cgpms, col0) if col0 not in Zv\
            else [c for c in Zv if Zv[c]==Zv[col0]]
        ancestors1 = retrieve_ancestors(cgpms, col1) if col1 not in Zv\
            else [c for c in Zv if Zv[c]==Zv[col1]]
        # Direct common ancestor implies dependent.
        if set.intersection(set(ancestors0), set(ancestors1)):
            return 1.
        # Dependent ancestors via variable partition at root, Zv.
        cc_ancestors0 = [Zv[i] for i in ancestors0 if i in Zv]
        cc_ancestors1 = [Zv[i] for i in ancestors1 if i in Zv]
        if set.intersection(set(cc_ancestors0), set(cc_ancestors1)):
            return 1.
        # No dependence.
        return 0.

    # --------------------------------------------------------------------------
    # Row similarity.

    def row_similarity(self, row0, row1, cols=None):
        if cols is None:
            cols = self.outputs
        views = set(self.view_for(c) for c in cols)
        return np.mean([v.Zr(row0)==v.Zr(row1) for v in views])

    def row_similarity_pairwise(self, cols=None):
        if cols is None:
            cols = self.outputs
        rowids = list(range(self.n_rows()))
        S = np.eye(len(rowids))
        for row0, row1 in itertools.combinations(rowids, 2):
            s = self.row_similarity(row0, row1, cols=cols)
            S[row0, row1] = S[row1, row0] = s
        return S

    # --------------------------------------------------------------------------
    # Relevance probability.

    def relevance_probability(
            self, rowid_target, rowid_query, col, hypotheticals=None):
        """Compute relevance probability of query rows for target row."""
        assert col in self.outputs
        # Retrieve the relevant view.
        view = self.view_for(col)
        # Select the hypothetical rows which are compatible with the view.
        hypotheticals = [r for r in [{d: h.get(d, np.nan) for d in view.dims} for h in hypotheticals] if not all(np.isnan(list(r.values())))] if hypotheticals else []
        # Produce hypothetical rowids.
        rowid_hypothetical = list(range(
            self.n_rows(), self.n_rows() + len(hypotheticals)))
        # Incorporate hypothetical rows.
        for rowid, query in zip(rowid_hypothetical, hypotheticals):
            for d in view.dims:
                self.X[d].append(query[d])
            view.incorporate(rowid, query)
        # Compute the relevance probability.
        rowid_all = rowid_query + rowid_hypothetical
        relevance = all(
            view.Zr(rowid_target) == view.Zr(rq)
            for rq in rowid_all
        ) if rowid_all else 0
        # Unincorporate hypothetical rows.
        for rowid in reversed(rowid_hypothetical):
            for d in view.dims:
                self.X[d].pop()
            view.unincorporate(rowid)
        return int(relevance)

    # --------------------------------------------------------------------------
    # Mutual information

    def mutual_information(self, col0, col1, constraints=None, T=None, N=None,
            progress=None):
        """Computes the mutual information MI(col0:col1|constraints).

        Mutual information with constraints can be of the form:
            - MI(X:Y|Z=z): CMI at a fixed conditioning value.
            - MI(X:Y|Z): expected CMI E_Z[MI(X:Y|Z)] under Z.
            - MI(X:Y|Z, W=w): expected CMI E_Z[MI(X:Y|Z,W=w)] under Z.

        This function supports all three forms. The CMI is computed under the
        posterior predictive joint distributions.

        Parameters
        ----------
        col0, col1 : list<int>
            Columns to comptue MI. If all columns in `col0` are equivalent
            to columns in `col` then entropy is returned, otherwise they must
            be disjoint and the CMI is returned
        constraints : list(tuple), optional
            A list of pairs (col, val) of observed values to condition on. If
            `val` is None, then `col` is marginalized over.
        T : int, optional.
            Number of samples to use in the outer (marginalization) estimator.
        N : int, optional.
            Number of samples to use in the inner Monte Carlo estimator.

        Returns
        -------
        mi : float
            A point estimate of the mutual information.

        Examples
        -------
        # Compute MI(X:Y)
        >>> State.mutual_information(col_x, col_y)
        # Compute MI(X:Y|Z=1)
        >>> State.mutual_information(col_x, col_y, {col_z: 1})
        # Compute MI(X:Y|W)
        >>> State.mutual_information(col_x, col_y, {col_w:None})
        # Compute MI(X:Y|Z=1, W)
        >>> State.mutual_information(col_x, col_y, {col_z: 1, col_w:None})
        """
        if constraints is None:
            constraints = dict()
        # Disallow duplicated variables in constraints and targets.
        if any(c in constraints for c in col0 + col1):
            raise ValueError('Target and constraints columns must be disjoint.')
        # Disallow duplicates in targets, except exact match (entropy).
        if any(c in col1 for c in col0) and set(col0) != set(col1):
            raise ValueError('Targets must match exactly or be disjoint.')
        # Partition the query into independent blocks.
        blocks = self._partition_mutual_information_query(
            col0, col1, constraints)
        return sum(
            self._compute_mutual_information(c0, c1, const, T, N, progress)
            for c0, c1, const in blocks
            if c0 and c1
        )

    def _compute_mutual_information(self, col0, col1, constraints, T=None,
            N=None, progress=None):
        N = N or 100
        T = T or 100
        # Partition constraints into equality (e) and marginalization (m) forms.
        e_constraints = {e:x for e,x in constraints.items() if x is not None}
        m_constraints = [e for e,x in constraints.items() if x is None]
        # Determine the estimator to use.
        estimator = self._compute_mi if set(col0) != set(col1) \
            else self._compute_entropy
        # No marginalization constraints.
        if not m_constraints:
            return estimator(col0, col1, constraints, N)
        # Compute CMI by Monte Carlo.
        def compute_one(i, sample):
            const = gu.merged(e_constraints, sample)
            m = estimator(col0, col1, const, N)
            if progress:
                self._progress(float(i)/T)
            return m
        if progress:
            self._progress(0./T)
        m_samples = self.simulate(None, m_constraints, N=T)
        mi = sum(compute_one(i, samp) for i, samp in enumerate(m_samples))
        return mi / float(T)

    def _compute_mi(self, col0, col1, constraints, N):
        samples = self.simulate(None, col0 + col1, constraints, None, N)
        PXY = self.logpdf_bulk(
            rowids=[-1]*N,
            targets_list=samples,
            constraints_list=[constraints]*N
        )
        PX = self.logpdf_bulk(
            rowids=[-1]*N,
            targets_list=[{c0: s[c0] for c0 in col0} for s in samples],
            constraints_list=[constraints]*N,
        )
        PY = self.logpdf_bulk(
            rowids=[-1]*N,
            targets_list=[{c1: s[c1] for c1 in col1} for s in samples],
            constraints_list=[constraints]*N,
        )
        return old_div((np.sum(PXY) - np.sum(PX) - np.sum(PY)), N)

    def _compute_entropy(self, col0, col1, constraints, N):
        assert set(col0) == set(col1)
        samples = self.simulate(-1, col0, constraints, None, N)
        PX = self.logpdf_bulk(
            rowids=[-1]*N,
            targets_list=[{c0: s[c0] for c0 in col0} for s in samples],
            constraints_list=[constraints]*N,
        )
        return old_div(-np.sum(PX), N)

    def _partition_mutual_information_query(self, col0, col1, constraints):
        cgpms = self.build_cgpms()
        var_to_cgpm = retrieve_variable_to_cgpm(cgpms)
        connected_components = retrieve_weakly_connected_components(cgpms)
        blocks = defaultdict(lambda: ([], [], {}))
        for variable in col0:
            component = connected_components[var_to_cgpm[variable]]
            blocks[component][0].append(variable)
        for variable in col1:
            component = connected_components[var_to_cgpm[variable]]
            blocks[component][1].append(variable)
        for variable in constraints:
            component = connected_components[var_to_cgpm[variable]]
            blocks[component][2][variable] = constraints[variable]
        return list(blocks.values())

    # --------------------------------------------------------------------------
    # Inference

    def transition(
            self, N=None, S=None, kernels=None, rowids=None,
            cols=None, views=None, progress=True, checkpoint=None):
        """Run targeted inference kernels.

        Parameters
        ----------
        N : int, optional
            Number of iterations to transition. Default 1.
        S : float, optional
            Number of seconds to transition. If both N and S set then min used.
        kernels : list<{'alpha', 'view_alphas', 'column_params', 'column_hypers'
            'rows', 'columns'}>, optional
            List of inference kernels to run in this transition. Default all.
        views, rows, cols : list<int>, optional
            View, row and column numbers to apply the kernels. Default all.
        checkpoint : int, optional
            Number of transitions between recording inference diagnostics
            from the latent state (such as logscore and row/column partitions).
            Defaults to no checkpointing.
        progress : boolean, optional
            Show a progress bar for number of target iterations or elapsed time.
        """
        # XXX Many combinations of the above kwargs will cause havoc.

        # Check columns exist, silently ignore non-existent columns.
        if cols and any(c not in self.outputs for c in cols):
            raise ValueError('Only CrossCat columns may be transitioned.')

        # Default order of crosscat kernels is important.
        _kernel_lookup = OrderedDict([
            ('structure_hypers',
                lambda : self.transition_structure_hypers()),
            ('view_structure_hypers',
                lambda : self.transition_view_structure_hypers(views=views, cols=cols)),
            ('column_params',
                lambda : self.transition_dim_params(cols=cols)),
            ('column_hypers',
                lambda : self.transition_dim_hypers(cols=cols)),
            ('rows',
                lambda : self.transition_view_rows(
                    views=views, cols=cols, rows=rowids)),
            ('columns' ,
                lambda : self.transition_dims(cols=cols)),
        ])

        # Run all kernels by default.
        if kernels is None:
            kernels = list(_kernel_lookup.keys())

        kernel_funcs = [_kernel_lookup[k] for k in kernels]
        assert kernel_funcs

        self._transition_generic(
            kernel_funcs, N=N, S=S, progress=progress, checkpoint=checkpoint)

    def transition_structure_hypers(self):
        self.crp.transition_hypers()
        self._increment_iterations('structure_hypers')

    def transition_view_structure_hypers(self, views=None, cols=None):
        if views is None:
            views = set(self.Zv(col) for col in cols) if cols else self.views
        for v in views:
            self.views[v].transition_crp_hypers()
        self._increment_iterations('view_structure_hypers')

    def transition_dim_params(self, cols=None):
        if cols is None:
            cols = self.outputs
        for c in cols:
            self.dim_for(c).transition_params()
        self._increment_iterations('column_params')

    def transition_dim_hypers(self, cols=None):
        if cols is None:
            cols = self.outputs
        for c in cols:
            self.dim_for(c).transition_hypers()
        self._increment_iterations('column_hypers')

    def transition_dim_grids(self, cols=None):
        if cols is None:
            cols = self.outputs
        for c in cols:
            self.dim_for(c).transition_hyper_grids(self.X[c])
        self._increment_iterations('column_grids')

    def transition_view_rows(self, views=None, rows=None, cols=None):
        if self.n_rows() == 1:
            return
        if views is None:
            views = set(self.Zv(col) for col in cols) if cols else self.views
        for v in views:
            self.views[v].transition_rows(rows=rows)
        self._increment_iterations('rows')

    def transition_dims(self, cols=None, m=1):
        if cols is None:
            cols = self.outputs
        cols = self.rng.permutation(cols)
        for c in cols:
            self._gibbs_transition_dim(c, m)
        self._increment_iterations('columns')

    def _transition_generic(
            self, kernels, N=None, S=None, progress=None, checkpoint=None):

        def _proportion_done(N, S, iters, start):
            if S is None:
                p_seconds = 0
            else:
                p_seconds = old_div((time.time() - start), S)
            if N is None:
                p_iters = 0
            else:
                p_iters = float(iters)/N
            return max(p_iters, p_seconds)

        if N is None and S is None:
            N = 1
        if progress is None:
            progress = True

        iters = 0
        start = time.time()

        while True and kernels:
            for kernel in kernels:
                p = _proportion_done(N, S, iters, start)
                if progress:
                    self._progress(p)
                if p >= 1.:
                    break
                kernel()
            else:
                iters += 1
                if checkpoint and (iters % checkpoint == 0):
                    self._increment_diagnostics()
                continue
            break

        if progress:
            print('\rCompleted: %d iterations in %f seconds.' % \
                (iters, time.time()-start))

    def _increment_iterations(self, kernel, N=1):
        previous = self.diagnostics['iterations'].get(kernel, 0)
        self.diagnostics['iterations'][kernel] = previous + N

    def _increment_diagnostics(self):
        self.diagnostics['logscore'].append(self.logpdf_score())
        self.diagnostics['structure_hypers'].append(self.crp.hypers)
        self.diagnostics['column_partition'].append(list(self.Zv().items()))

    def _progress(self, percentage):
        tu.progress(percentage, sys.stdout)


    # --------------------------------------------------------------------------
    # Helpers

    def data_array(self):
        """Return dataset as a numpy array."""
        return np.asarray(list(self.X.values())).T

    def n_rows(self):
        """Number of incorporated rows."""
        return len(self.X[self.outputs[0]])

    def n_cols(self):
        """Number of incorporated columns."""
        return len(self.outputs)

    def cctypes(self):
        """DistributionGpm name of each Dim."""
        return [d.name() for d in self.dims()]

    def distargs(self):
        """DistributionGpm distargs of each Dim."""
        return [d.get_distargs() for d in self.dims()]

    # --------------------------------------------------------------------------
    # Helpers for the outputs

    def has_output(self, col):
        """Optimized lookup checking if state has output."""
        return col in self.outputs_set

    def set_outputs(self, outputs):
        """Update the outputs of the states."""
        self.outputs = list(outputs)
        self.outputs_set = set(self.outputs)

    # --------------------------------------------------------------------------
    # Internal CRP utils.

    def Nv(self, v=None):
        Nv = self.crp.clusters[0].counts
        return Nv[v] if v is not None else Nv.copy()

    def Zv(self, c=None):
        Zv = self.crp.clusters[0].data
        return Zv[c] if c is not None else Zv.copy()

    # --------------------------------------------------------------------------
    # Accessors

    def dim_for(self, c):
        return self.view_for(c).dims[c]

    def dims(self):
        return [self.view_for(c).dims[c] for c in self.outputs]

    def view_for(self, c):
        return self.views[self.Zv(c)]

    # --------------------------------------------------------------------------
    # Inference helpers.

    def _dim_is_member(self, view, dim):
        """Is dim a member of view?"""
        return view is not None and dim.index in view.dims

    # Compute probability of dim data under view partition.
    def _dim_get_data_logp(self, view, dim):
        # collasped   member  reassign
        # 0           0       1
        # 0           1       0
        # 1           0       1
        # 1           0       1
        # implies reassign = collapsed or (not member)
        reassign = dim.is_collapsed() or not self._dim_is_member(view, dim)
        logp = view.incorporate_dim(dim, reassign=reassign)
        view.unincorporate_dim(dim)
        return logp

    def _dim_get_proposal(self, view, dim):
        """Get a dim object propose to the view."""
        # If collapsed dim, reuse the dim object. Otherwise uncollapsed dim,
        # create copy to preserve uncollapsed state (can optimize).
        if dim.is_collapsed() or self._dim_is_member(view, dim):
            return dim
        return copy.deepcopy(dim)

    def _gibbs_transition_dim(self, col, m):
        """Gibbs on col assignment to Views, with m auxiliary parameters"""
        # XXX Disable col transitions if \exists conditional model anywhere.
        if any(d.is_conditional() for d in self.dims()):
            raise ValueError(
                'Cannot transition columns with conditional dims.')
        if self.Cd:
            raise ValueError(
                'Cannot transition columns with dependence constraint, '
                'use State.transition_lovecat.')

        # Current dim object and view index.
        dim = self.dim_for(col)

        # Compute logp of the dim under existing views.
        dims_proposal = [
            self._dim_get_proposal(self.views[view], dim)
            for view in self.views
        ]
        logp_data = [
            self._dim_get_data_logp(self.views[view], dim)
            for (view, dim) in zip(self.views, dims_proposal)
        ]

        # Compute logp of the dim under auxiliary views.
        tables = self.crp.clusters[0].gibbs_tables(col, m=m)
        t_aux = tables[len(self.views):]
        dims_proposal_aux = [self._dim_get_proposal(None, dim) for _t in t_aux]
        views_proposal_aux = [
            View(self.X, outputs=[self.crp_id_view + t], rng=self.rng)
            for t in t_aux
        ]
        logp_data_aux = [
            self._dim_get_data_logp(view, dim)
            for (view, dim) in zip(views_proposal_aux, dims_proposal_aux)
        ]

        # Extend structures with auxiliary proposals.
        dims_proposal.extend(dims_proposal_aux)
        logp_data.extend(logp_data_aux)

        # Compute the CRP probabilities of each view.
        logp_crp = self.crp.clusters[0].gibbs_logps(col, m=m)
        assert len(logp_data) == len(logp_crp)

        # Overall view probabilities.
        logp_views = np.add(logp_data, logp_crp)

        # Enforce independence constraints.
        avoid = [a for p in self.Ci if col in p for a in p if a != col]
        for a in avoid:
            index = list(self.views.keys()).index(self.Zv(a))
            logp_views[index] = float('-inf')

        # Draw a new view.
        assert len(tables) == len(logp_views)
        draw = gu.log_pflip(logp_views, rng=self.rng)
        v_sampled = tables[draw]
        v_current = self.Zv(col)

        # Migrate dimension to a new view if necessary.
        if v_current != v_sampled:
            # If migrating dim to an aux view, add it to the state.
            if v_sampled > max(self.views):
                view_aux = views_proposal_aux[draw-len(self.views)]
                self._append_view(view_aux, v_sampled)
            self._migrate_dim(v_current, v_sampled, dims_proposal[draw])
        else:
            self.views[v_current].incorporate_dim(
                dims_proposal[draw],
                reassign=dims_proposal[draw].is_collapsed(),
            )

        self._check_partitions()

    def _migrate_dim(self, v_a, v_b, dim, reassign=None):
        # If `reassign`, then the row partition in `dim` will be force
        # reassigned; if False, then dim.clusters is expected to already match
        # that of view. By default, only collapsed columns will be reassign, and
        # uncollapsed columns (so that the user can specify the uncollasped
        # cluster parameters without having the migration overwrite them).
        if reassign is None:
            reassign = dim.is_collapsed()
        # XXX Even though dim might not be a member of view v_a, the CRP gpm
        # which stores the counts has not been updated to reflect the removal of
        # dim from v_a. Therefore, we check whether CRP has v_a as a singleton.
        delete = self.Nv(v_a) == 1
        if dim.index in self.views[v_a].dims:
            self.views[v_a].unincorporate_dim(dim)
        self.views[v_b].incorporate_dim(dim, reassign=reassign)
        # CRP accounting.
        self.crp.unincorporate(dim.index)
        self.crp.incorporate(dim.index, {self.crp_id: v_b}, {-1:0})
        # Delete empty view?
        if delete:
            self._delete_view(v_a)

    def _delete_view(self, v):
        assert v not in self.crp.clusters[0].counts
        del self.views[v]

    def _append_view(self, view, identity):
        """Append a view and return and its index."""
        assert len(view.dims) == 0
        self.views[identity] = view

    def hypothetical(self, rowid):
        if rowid is not None:
            return not 0 <= rowid < self.n_rows()
        return True

    # --------------------------------------------------------------------------
    # Data structure invariants.

    def _check_partitions(self):
        if not cu.check_env_debug():
            return
        assert self.crp.hypers['alpha'] > 0.
        assert all(len(self.views[v].dims) == self.crp.clusters[0].counts[v]
                for v in self.views)
        # All outputs should be in the dataset keys.
        assert all([c in list(self.X.keys()) for c in self.outputs])
        # Zv and dims should match n_cols.
        assert sorted(self.Zv().keys()) == sorted(self.outputs)
        assert len(self.Zv()) == self.n_cols()
        assert len(self.dims()) == self.n_cols()
        # Nv should account for each column.
        assert sum(self.Nv().values()) == self.n_cols()
        # Nv should have an entry for each view.
        # assert len(self.Nv_list()) == max(self.Zv.values())+1
        for v in self.views:
            assert len(self.views[v].dims) == self.Nv(v)
            self.views[v]._check_partitions()
        # Dependence constraints.
        assert vu.validate_crp_constrained_partition(
            [self.Zv(c) for c in self.outputs], self.Cd, self.Ci,
            self.Rd, self.Ri)