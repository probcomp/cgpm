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

from __future__ import absolute_import
from builtins import zip
from builtins import range
import itertools

from math import isnan

import numpy as np

from cgpm.cgpm import CGpm
from cgpm.mixtures.dim import Dim
from cgpm.network.importance import ImportanceNetwork
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils.config import cctype_class
from cgpm.utils.general import merged


class View(CGpm):
    """CGpm represnting a multivariate Dirichlet process mixture of CGpms."""

    def __init__(
            self, X, outputs=None, inputs=None, alpha=None,
            cctypes=None, distargs=None, hypers=None, Zr=None, rng=None):
        """View constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data and optional row partition.

        Parameters
        ----------
        X : dict{int:list}
            Dataset, where the cell `X[outputs[i]][rowid]` contains the value
            for column outputs[i] and rowd index `rowid`. All rows are
            incorporated by default.
        outputs : list<int>
            List of output variables. The first item is mandatory, corresponding
            to the token of the exposed cluster. outputs[1:] are the observable
            output variables.
        inputs : list<int>
            Currently disabled.
        alpha : float, optional.
            Concentration parameter for row CRP.
        cctypes : list<str>, optional.
            A `len(outputs[1:])` list of cctypes, see `utils.config` for names.
        distargs : list<str>, optional.
            A `len(outputs[1:])` list of distargs.
        hypers : list<dict>, optional.
            A `len(outputs[1:])` list of hyperparameters.
        Zr : list<int>, optional.
            Row partition, where `Zr[rowid]` is the cluster identity of rowid.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        # -- Seed --------------------------------------------------------------
        self.rng = gu.gen_rng() if rng is None else rng

        # -- Inputs ------------------------------------------------------------
        if inputs:
            raise ValueError('View does not accept inputs.')
        self.inputs = []

        # -- Dataset -----------------------------------------------------------
        self.X = X

        # -- Outputs -----------------------------------------------------------
        if len(outputs) < 1:
            raise ValueError('View needs at least one output.')
        if len(outputs) > 1:
            if not distargs:
                distargs = [None] * len(cctypes)
            if not hypers:
                hypers = [None] * len(cctypes)
            assert len(outputs[1:])==len(cctypes)
            assert len(distargs) == len(cctypes)
            assert len(hypers) == len(cctypes)
        self.outputs = list(outputs)

        # -- Row CRP -----------------------------------------------------------
        self.crp = Dim(
            outputs=[self.outputs[0]],
            inputs=[-1],
            cctype='crp',
            hypers=None if alpha is None else {'alpha': alpha},
            rng=self.rng
        )
        n_rows = len(self.X[list(self.X.keys())[0]])
        self.crp.transition_hyper_grids([1]*n_rows)
        if Zr is None:
            for i in range(n_rows):
                s = self.crp.simulate(i, [self.outputs[0]], None, {-1:0})
                self.crp.incorporate(i, s, {-1:0})
        else:
            for i, z in enumerate(Zr):
                self.crp.incorporate(i, {self.outputs[0]: z}, {-1:0})

        # -- Dimensions --------------------------------------------------------
        self.dims = dict()
        for i, c in enumerate(self.outputs[1:]):
            # Prepare inputs for dim, if necessary.
            dim_inputs = []
            if distargs[i] is not None and 'inputs' in distargs[i]:
                dim_inputs = distargs[i]['inputs']['indexes']
            dim_inputs = [self.outputs[0]] + dim_inputs
            # Construct the Dim.
            dim = Dim(
                outputs=[c],
                inputs=dim_inputs,
                cctype=cctypes[i],
                hypers=hypers[i],
                distargs=distargs[i],
                rng=self.rng
            )
            dim.transition_hyper_grids(self.X[c])
            self.incorporate_dim(dim)

        # -- Validation --------------------------------------------------------
        self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, dim, reassign=True):
        """Incorporate dim into View. If not reassign, partition should match."""
        dim.inputs[0] = self.outputs[0]
        if reassign:
            self._bulk_incorporate(dim)
        self.dims[dim.index] = dim
        self.outputs = self.outputs[:1] + list(self.dims.keys())
        return dim.logpdf_score()

    def unincorporate_dim(self, dim):
        """Remove dim from this View (does not modify)."""
        del self.dims[dim.index]
        self.outputs = self.outputs[:1] + list(self.dims.keys())
        return dim.logpdf_score()

    def incorporate(self, rowid, observation, inputs=None):
        """Incorporate an observation into the View.

        Parameters
        ----------
        rowid : int
            Fresh, non-negative rowid.
        observation : dict{output:val}
            Keys of the observation must exactly be the output (Github #89).
            Optionally, use {self.outputs[0]: k} to specify the latent cluster
            assignment of rowid. The cluster is an observation variable since
            View has a generative model for k, unlike Dim which requires k as
            inputs.
        """
        k = observation.get(self.outputs[0], 0)
        self.crp.incorporate(rowid, {self.outputs[0]: k}, {-1: 0})
        for d in self.dims:
            self.dims[d].incorporate(
                rowid,
                observation={d: observation[d]},
                inputs=self._get_input_values(rowid, self.dims[d], k))
        # If the user did not specify a cluster assignment, sample one.
        if self.outputs[0] not in observation:
            self.transition_rows(rows=[rowid])

    def unincorporate(self, rowid):
        # Unincorporate from dims.
        for dim in self.dims.values():
            dim.unincorporate(rowid)
        # Account.
        k = self.Zr(rowid)
        self.crp.unincorporate(rowid)
        if k not in self.Nk():
            for dim in self.dims.values():
                del dim.clusters[k]     # XXX Abstract me!

    # XXX Major hack to force values of NaN cells in incorporated rowids.
    def force_cell(self, rowid, observation):
        k = self.Zr(rowid)
        for d in observation:
            self.dims[d].unincorporate(rowid)
            inputs = self._get_input_values(rowid, self.dims[d], k)
            self.dims[d].incorporate(rowid, {d: observation[d]}, inputs)

    # --------------------------------------------------------------------------
    # Update schema.

    def update_cctype(self, col, cctype, distargs=None):
        """Update the distribution type of self.dims[col] to cctype."""
        if distargs is None:
            distargs = {}
        distargs_dim = dict(distargs)
        inputs = []
        # XXX Horrid hack.
        if cctype_class(cctype).is_conditional():
            inputs = distargs_dim.get('inputs', [
                d for d in sorted(self.dims)
                if d != col and not self.dims[d].is_conditional()
            ])
            if len(self.dims) == 0 or len(inputs) == 0:
                raise ValueError('No inputs for conditional dimension.')
            distargs_dim['inputs'] = {
                'indexes' : inputs,
                'stattypes': [self.dims[i].cctype for i in inputs],
                'statargs': [self.dims[i].get_distargs() for i in inputs]
            }
        D_old = self.dims[col]
        D_new = Dim(
            outputs=[col], inputs=[self.outputs[0]]+inputs,
            cctype=cctype, distargs=distargs_dim, rng=self.rng)
        self.unincorporate_dim(D_old)
        self.incorporate_dim(D_new)

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N):
        for _ in range(N):
            self.transition_rows()
            self.transition_crp_alpha()
            self.transition_dim_hypers()

    def transition_crp_alpha(self):
        self.crp.transition_hypers()
        self.crp.transition_hypers()

    def transition_dim_hypers(self, cols=None):
        if cols is None:
            cols = list(self.dims.keys())
        for c in cols:
            self.dims[c].transition_hypers()

    def transition_dim_grids(self, cols=None):
        if cols is None:
            cols = list(self.dims.keys())
        for c in cols:
            self.dims[c].transition_hyper_grids(self.X[c])

    def transition_rows(self, rows=None):
        if rows is None:
            rows = list(self.Zr().keys())
        rows = self.rng.permutation(rows)
        for rowid in rows:
            self._gibbs_transition_row(rowid)

    # --------------------------------------------------------------------------
    # logscore.

    def logpdf_likelihood(self):
        """Compute the logpdf of the observations only."""
        logp_dims = [dim.logpdf_score() for dim in self.dims.values()]
        return sum(logp_dims)

    def logpdf_prior(self):
        logp_crp = self.crp.logpdf_score()
        return logp_crp

    def logpdf_score(self):
        """Compute the marginal logpdf CRP assignment and data."""
        lp_prior = self.logpdf_prior()
        lp_likelihood = self.logpdf_likelihood()
        return lp_prior + lp_likelihood

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        # As discussed in https://github.com/probcomp/cgpm/issues/116 for an
        # observed rowid, we synthetize a new hypothetical row which is
        # identical (in terms of observed and latent values) to the observed
        # rowid. In this version of the implementation, the user may not
        # override any non-null values in the observed rowid
        # (_populate_constraints returns an error in this case). A user should
        # either (i) use another rowid, since overriding existing values in the
        # observed rowid no longer specifies that rowid, or (ii) use some
        # sequence of incorporate/unicorporate depending on their query.
        constraints = self._populate_constraints(rowid, targets, constraints)
        if not self.hypothetical(rowid):
            rowid = None
        # Prepare the importance network.
        network = self.build_network()
        if self.outputs[0] in constraints:
            # Condition on the cluster assignment.
            # p(xT|xC,z=k)                      computed directly by network.
            return network.logpdf(rowid, targets, constraints, inputs)
        elif self.outputs[0] in targets:
            # Query the cluster assignment.
            # p(z=k,xT|xC)
            # = p(z=k,xT,xC) / p(xC)            Bayes rule
            # = p(z=k)p(xT,xC|z=k) / p(xC)      chain rule on numerator
            # The terms are then:
            # p(z=k)                            lp_cluster
            # p(xT,xC|z=k)                      lp_numer
            # p(xC)                             lp_denom
            k = targets[self.outputs[0]]
            constraints_z = {self.outputs[0]: k}
            targets_nz = {c: targets[c] for c in targets if c != self.outputs[0]}
            targets_numer = merged(targets_nz, constraints)
            lp_cluster = network.logpdf(rowid, constraints_z, inputs)
            lp_numer = \
                network.logpdf(rowid, targets_numer, constraints_z, inputs) \
                if targets_numer else 0
            lp_denom = self.logpdf(rowid, constraints) if constraints else 0
            return (lp_cluster + lp_numer) - lp_denom
        else:
            # Marginalize over cluster assignment by enumeration.
            # Let K be a list of values for the support of z:
            # P(xT|xC)
            # = \sum_k p(xT|z=k,xC)p(z=k|xC)            marginalization
            # Now consider p(z=k|xC) \propto p(z=k,xC)  Bayes rule
            # p(z=K[i],xC)                              lp_constraints_unorm[i]
            # p(z=K[i]|xC)                              lp_constraints[i]
            # p(xT|z=K[i],xC)                           lp_targets[i]
            K = self.crp.clusters[0].gibbs_tables(-1)
            constraints = [merged(constraints, {self.outputs[0]: k}) for k in K]
            lp_constraints_unorm = [network.logpdf(rowid, const, None, inputs)
                for const in constraints]
            lp_constraints = gu.log_normalize(lp_constraints_unorm)
            lp_targets = [network.logpdf(rowid, targets, const, inputs)
                for const in constraints]
            return gu.logsumexp(np.add(lp_constraints, lp_targets))

    # --------------------------------------------------------------------------
    # simulate

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        # Refer to comment in logpdf.
        constraints = self._populate_constraints(rowid, targets, constraints)
        if not self.hypothetical(rowid):
            rowid = None
        network = self.build_network()
        # Condition on the cluster assignment.
        if self.outputs[0] in constraints:
            return network.simulate(rowid, targets, constraints, inputs, N)
        # Determine how many samples to return.
        unwrap_result = N is None
        if unwrap_result:
            N = 1
        # Expose cluster assignments to the samples?
        exposed = self.outputs[0] in targets
        if exposed:
            targets = [q for q in targets if q != self.outputs[0]]
        # Weight clusters by probability of constraints in each cluster.
        K = self.crp.clusters[0].gibbs_tables(-1)
        constr2 = [merged(constraints, {self.outputs[0]: k}) for k in K]
        lp_constraints_unorm = [network.logpdf(rowid, ev) for ev in constr2]
        # Find number of samples in each cluster.
        Ks = gu.log_pflip(lp_constraints_unorm, array=K, size=N, rng=self.rng)
        counts = {k:n for k, n in enumerate(np.bincount(Ks)) if n > 0}
        # Add the cluster assignment to the constraints and sample the rest.
        constr3 = {k: merged(constraints, {self.outputs[0]: k}) for k in counts}
        samples = [network.simulate(rowid, targets, constr3[k], inputs, counts[k])
            for k in counts]
        # If cluster assignments are exposed, append them to the samples.
        if exposed:
            samples = [[merged(l, {self.outputs[0]: k}) for l in s]
                for s, k in zip(samples, counts)]
        # Return 1 sample if N is None, otherwise a list.
        result = list(itertools.chain.from_iterable(samples))
        return result[0] if unwrap_result else result


    # --------------------------------------------------------------------------
    # Internal simulate/logpdf helpers

    def relevance_probability(self, rowid_target, rowid_query, col):
        """Compute probability of rows in same cluster."""
        if col not in self.outputs:
            raise ValueError('Unknown column: %s' % (col,))
        from .relevance import relevance_probability
        return relevance_probability(self, rowid_target, rowid_query)

    # --------------------------------------------------------------------------
    # Internal simulate/logpdf helpers

    def build_network(self):
        return ImportanceNetwork(
            cgpms=[self.crp.clusters[0]] + list(self.dims.values()),
            accuracy=1,
            rng=self.rng)

    # --------------------------------------------------------------------------
    # Internal row transition.

    def _gibbs_transition_row(self, rowid):
        # Probability of row crp assignment to each cluster.
        K = self.crp.clusters[0].gibbs_tables(rowid)
        logp_crp = self.crp.clusters[0].gibbs_logps(rowid)
        # Probability of row data in each cluster.
        logp_data = self._logpdf_row_gibbs(rowid, K)
        assert len(logp_data) == len(logp_crp)
        # Sample new cluster.
        p_cluster = np.add(logp_data, logp_crp)
        z_b = gu.log_pflip(p_cluster, array=K, rng=self.rng)
        # Migrate the row.
        if self.Zr(rowid) != z_b:
            self._migrate_row(rowid, z_b)
        self._check_partitions()

    def _logpdf_row_gibbs(self, rowid, K):
        return [sum([self._logpdf_cell_gibbs(rowid, dim, k)
            for dim in self.dims.values()]) for k in K]

    def _logpdf_cell_gibbs(self, rowid, dim, k):
        targets = {dim.index: self.X[dim.index][rowid]}
        inputs = self._get_input_values(rowid, dim, k)
        # If rowid in cluster k then unincorporate then compute predictive.
        if self.Zr(rowid) == k:
            dim.unincorporate(rowid)
            logp = dim.logpdf(rowid, targets, None, inputs)
            dim.incorporate(rowid, targets, inputs)
        else:
            logp = dim.logpdf(rowid, targets, None, inputs)
        return logp

    def _migrate_row(self, rowid, k):
        self.unincorporate(rowid)
        observation = merged(
            {d: self.X[d][rowid] for d in self.dims},
            {self.outputs[0]: k})
        self.incorporate(rowid, observation)

    # --------------------------------------------------------------------------
    # Internal crp utils.

    def alpha(self):
        return self.crp.hypers['alpha']

    def Nk(self, k=None):
        Nk = self.crp.clusters[0].counts
        return Nk[k] if k is not None else Nk

    def Zr(self, rowid=None):
        Zr = self.crp.clusters[0].data
        return Zr[rowid] if rowid is not None else Zr

    # --------------------------------------------------------------------------
    # Internal query utils.


    def n_rows(self):
        return len(self.Zr())

    def hypothetical(self, rowid):
        if (rowid is not None):
            return not (0 <= rowid < len(self.Zr()))
        return True

    def _populate_constraints(self, rowid, targets, constraints):
        """Loads constraints from the dataset."""
        if constraints is None:
            constraints = {}
        self._validate_cgpm_query(rowid, targets, constraints)
        # If the rowid is hypothetical, just return.
        if self.hypothetical(rowid):
            return constraints
        # Retrieve all values for this rowid not in targets or constraints.
        data = {
            c: self.X[c][rowid]
            for c in self.outputs[1:]
            if \
                c not in targets \
                and c not in constraints \
                and not isnan(self.X[c][rowid])
        }
        # Add the cluster assignment.
        data[self.outputs[0]] = self.Zr(rowid)

        return merged(constraints, data)

    def _get_input_values(self, rowid, dim, k):
        """Prepare the inputs for a Dim logpdf or simulate query."""
        inputs = {i: self.X[i][rowid] for i in dim.inputs[1:]}
        cluster = {self.outputs[0]: k}
        return merged(inputs, cluster)

    def _bulk_incorporate(self, dim):
        # XXX Major hack! We should really be creating new Dim objects.
        dim.clusters = {}   # Mapping of cluster k to the object.
        dim.Zr = {}         # Mapping of non-nan rowids to cluster k.
        dim.Zi = {}         # Mapping of nan rowids to cluster k.
        dim.aux_model = dim.create_aux_model()
        for rowid, k in self.Zr().items():
            observation = {dim.index: self.X[dim.index][rowid]}
            inputs = self._get_input_values(rowid, dim, k)
            dim.incorporate(rowid, observation, inputs)
        assert merged(dim.Zr, dim.Zi) == self.Zr()
        dim.transition_params()

    def _validate_cgpm_query(self, rowid, targets, constraints):
        # Is the query simulate or logpdf?
        simulate = isinstance(targets, (list, tuple))
        # Disallow duplicated target cols.
        if simulate and len(set(targets)) != len(targets):
            raise ValueError('Columns in targets must be unique.')
        # Disallow overlap between targets and constraints.
        if len(set.intersection(set(targets), set(constraints))) > 0:
            raise ValueError('Targets and constraints must be disjoint.')
        # No further check.
        if self.hypothetical(rowid):
            return
        # Cannot constrain the cluster of observed rowid; unincorporate first.
        if self.outputs[0] in targets or self.outputs[0] in constraints:
            raise ValueError('Cannot constrain cluster of an observed rowid.')
        # Disallow constraints constraining/disagreeing with observed cells.
        def good_constraints(rowid, e):
            return \
                e not in self.outputs\
                or np.isnan(self.X[e][rowid]) \
                or np.allclose(self.X[e][rowid], constraints[e])
        if any(not good_constraints(rowid, e) for e in constraints):
            raise ValueError('Cannot use observed cell in constraints.')
        # The next check is enforced at the level of State not View.
        # Disallow query constraining observed cells (XXX logpdf, not simulate)
        # if not simulate and any(not np.isnan(self.X[q][rowid]) for q in query):
        #     raise ValueError('Cannot constrain observed cell in query.')


    # --------------------------------------------------------------------------
    # Data structure invariants.

    def _check_partitions(self):
        if not cu.check_env_debug():
            return
        # For debugging only.
        assert self.alpha() > 0.
        # Check that the number of dims actually assigned to the view
        # matches the count in Nv.
        Zr = self.Zr()
        Nk = self.Nk()
        rowids = list(range(self.n_rows()))
        assert set(Zr.keys()) == set(rowids)
        assert set(Zr.values()) == set(Nk)
        for i, dim in self.dims.items():
            # Assert first output is first input of the Dim.
            assert self.outputs[0] == dim.inputs[0]
            # Assert length of dataset is the same as rowids.
            assert len(self.X[i]) == len(rowids)
            # Ensure number of clusters in each dim in views[v]
            # is the same and as described in the view (K, Nk).
            assignments = merged(dim.Zr, dim.Zi)
            assert assignments == Zr
            assert set(assignments.values()) == set(Nk.keys())
            all_ks = list(dim.clusters.keys()) + list(dim.Zi.values())
            assert set(all_ks) == set(Nk.keys())
            for k in dim.clusters:
                # Law of conservation of rowids.
                rowids_k = [r for r in rowids if Zr[r]==k]
                cols = [dim.index]
                if dim.is_conditional():
                    cols.extend(dim.inputs[1:])
                data = [[self.X[c][r] for c in cols] for r in rowids_k]
                rowids_nan = np.any(np.isnan(data), axis=1) if data else []
                assert (dim.clusters[k].N + np.sum(rowids_nan) == Nk[k])

    # --------------------------------------------------------------------------
    # Metadata

    def to_metadata(self):
        metadata = dict()

        # Dataset.
        metadata['X'] = self.X
        metadata['outputs'] = self.outputs

        # View partition data.
        rowids = sorted(self.Zr().keys())
        metadata['Zr'] = [self.Zr(i) for i in rowids]
        metadata['alpha'] = self.alpha()

        # Column data.
        metadata['cctypes'] = []
        metadata['hypers'] = []
        metadata['distargs'] = []
        metadata['suffstats'] = []
        for c in self.outputs[1:]:
            metadata['cctypes'].append(self.dims[c].cctype)
            metadata['hypers'].append(self.dims[c].hypers)
            metadata['distargs'].append(self.dims[c].distargs)
            metadata['suffstats'].append(self.dims[c].get_suffstats().items())

        # Factory data.
        metadata['factory'] = ('cgpm.mixtures.view', 'View')

        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        return cls(
            metadata.get('X'),
            outputs=metadata.get('outputs', None),
            inputs=metadata.get('inputs', None),
            alpha=metadata.get('alpha', None),
            cctypes=metadata.get('cctypes', None),
            distargs=metadata.get('distargs', None),
            hypers=metadata.get('hypers', None),
            Zr=metadata.get('Zr', None),
            rng=rng)
