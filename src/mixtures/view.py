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

import itertools

from math import isnan

import numpy as np

from gpmcc.cgpm import CGpm
from gpmcc.mixtures.dim import Dim
from gpmcc.network.importance import ImportanceNetwork
from gpmcc.utils import general as gu
from gpmcc.utils.config import cctype_class
from gpmcc.utils.general import merged


class View(CGpm):
    """CGpm represnting a multivariate Dirichlet process mixture of CGpms."""

    def __init__(self, X, outputs=None, inputs=None, alpha=None,
            cctypes=None, distargs=None, hypers=None, Zr=None, rng=None):
        """View constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data and optional row partition.

        Parameters
        ----------
        X : np.ndarray
            Global dataset of dimension N x D, structured in the form
            X[outputs[i]][rowid] for i>1. All rows are incorporated by default.
        outputs : list<int>
            List of output variables. The first item is mandatory, corresponding
            to the token of the exposed cluster. outputs[1:] are the observable
            output variables.
        inputs : list<int>
            Currently disabled.
        alpha : float, optional.
            Concentration parameter for row CRP.
        cctypes : list<str>, optional.
            A `len(outputs[1:]`) list of cctypes, see `utils.config` for names.
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
            assert len(outputs[1:])==len(cctypes)==len(distargs)==len(hypers)
        self.outputs = outputs

        # -- Row CRP -----------------------------------------------------------
        crp_alpha = None if alpha is None else {'alpha': alpha}
        self.crp = Dim(
            [self.outputs[0]], [-1], cctype='crp', hypers=crp_alpha,
            rng=self.rng)
        self.crp.transition_hyper_grids([1]*self.n_rows())
        if Zr is None:
            for i in xrange(self.n_rows()):
                s = self.crp.simulate(i, [self.outputs[0]], {-1:0})
                self.crp.incorporate(i, s, {-1:0})
        else:
            for i, z in enumerate(Zr):
                self.crp.incorporate(i, {self.outputs[0]: z}, {-1:0})

        # -- Dimensions --------------------------------------------------------
        self.dims = dict()
        for i, c in enumerate(self.outputs[1:]):
            dim = Dim(
                outputs=[c], inputs=[self.outputs[0]], cctype=cctypes[i],
                hypers=hypers[i], distargs=distargs[i], rng=self.rng)
            dim.transition_hyper_grids(self.X[c])
            if dim.is_conditional():
                raise ValueError('Use incorporate for conditional dims.')
            self.incorporate_dim(dim)

        # -- Validation --------------------------------------------------------
        self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, dim, reassign=True):
        """Incorporate dim into View. If not reassign, partition should match."""
        dim.inputs[0] = self.outputs[0]
        if reassign:
            distargs = self._prepare_incorporate(dim.cctype)
            dim.distargs.update(distargs)
            self._bulk_incorporate(dim)
        self.dims[dim.index] = dim
        self.outputs = self.outputs[:1] + self.dims.keys()
        return dim.logpdf_score()

    def unincorporate_dim(self, dim):
        """Remove dim from this View (does not modify)."""
        del self.dims[dim.index]
        self.outputs = self.outputs[:1] + self.dims.keys()
        return dim.logpdf_score()

    def incorporate(self, rowid, query, evidence=None):
        """Incorporate an observation into the View.

        Parameters
        ----------
        rowid : int
            Fresh, non-negative rowid.
        query : dict{output:val}
            Keys of the query must exactly be the output (Github issue 89).
            Optionally, use {self.outputs[0]: k} for latent cluster assignment
            of rowid. The cluster is a query variable since View
            has a generative model for k, unlike Dim which takes k as evidence.
        """
        k = query.get(self.outputs[0], 0)
        transition = [rowid] if k is None else []
        self.crp.incorporate(rowid, {self.outputs[0]: k}, {-1: 0})
        for d in self.dims:
            self.dims[d].incorporate(
                rowid,
                query={d: query[d]},
                evidence=self._get_evidence(rowid, self.dims[d], k))
        self.transition_rows(rows=transition)

    def unincorporate(self, rowid):
        # Unincorporate from dims.
        for dim in self.dims.itervalues():
            dim.unincorporate(rowid)
        # Account.
        k = self.Zr(rowid)
        self.crp.unincorporate(rowid)
        if k not in self.Nk():
            for dim in self.dims.itervalues():
                del dim.clusters[k]     # XXX Abstract me!

    # --------------------------------------------------------------------------
    # Update schema.

    def update_cctype(self, col, cctype, distargs=None):
        """Update the distribution type of self.dims[col] to cctype."""
        if distargs is None:
            distargs = {}
        inputs = []
        local_distargs = self._prepare_incorporate(cctype)
        if cctype_class(cctype).is_conditional():
            inputs = self._unconditional_dims()
            # Remove self-refrences when updating unconditional to conditional.
            if col in inputs:
                me = inputs.index(col)
                del local_distargs['cctypes'][me]
                del local_distargs['ccargs'][me]
                del inputs[me]
        inputs = [self.outputs[0]] + inputs
        distargs.update(local_distargs)
        D_old = self.dims[col]
        D_new = Dim(
            outputs=[col], inputs=inputs, cctype=cctype,
            distargs=distargs, rng=self.rng)
        self.unincorporate_dim(D_old)
        self.incorporate_dim(D_new)

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N):
        for _ in xrange(N):
            self.transition_rows()
            self.transition_crp_alpha()
            self.transition_dim_hypers()

    def transition_crp_alpha(self):
        self.crp.transition_hypers()
        self.crp.transition_hypers()

    def transition_dim_hypers(self, cols=None):
        if cols is None:
            cols = self.dims.keys()
        for c in cols:
            self.dims[c].transition_hypers()

    def transition_rows(self, rows=None):
        if rows is None:
            rows = self.Zr().keys()
        for rowid in rows:
            self._gibbs_transition_row(rowid)

    # --------------------------------------------------------------------------
    # logscore.

    def logpdf_score(self):
        """Compute the marginal logpdf CRP assignment and data."""
        logp_crp = self.crp.logpdf_score()
        logp_dims = [dim.logpdf_score() for dim in self.dims.itervalues()]
        return logp_crp + sum(logp_dims)

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence=None):
        # Algorithm.
        # P(xQ|xE) = \sum_z p(xQ|z,xE)p(z|xE)       marginalization
        # Now consider p(z|xE) \propto p(z)p(xE|z)  Bayes rule
        # p(z)p(xE|z)                               logp_evidence_unorm
        # p(z|xE)                                   logp_evidence
        # p(xQ|z)                                   logp_query
        evidence = self._populate_evidence(rowid, query, evidence)
        network = self.build_network()
        # Condition on cluster.
        if self.outputs[0] in evidence:
            # XXX DETERMINE ME!
            if not self.hypothetical(rowid): rowid = -1
            return network.logpdf(rowid, query, evidence)
        # Marginalize over clusters.
        K = self.crp.clusters[0].gibbs_tables(-1)
        evidences = [merged(evidence, {self.outputs[0]: k}) for k in K]
        lp_evidence_unorm = [network.logpdf(rowid, ev) for ev in evidences]
        lp_evidence = gu.log_normalize(lp_evidence_unorm)
        lp_query = [network.logpdf(rowid, query, ev) for ev in evidences]
        return gu.logsumexp(np.add(lp_evidence, lp_query))

    # --------------------------------------------------------------------------
    # simulate

    def simulate(self, rowid, query, evidence=None, N=None):
        evidence = self._populate_evidence(rowid, query, evidence)
        network = self.build_network()
        # Condition on cluster.
        if self.outputs[0] in evidence:
            # XXX DETERMINE ME!
            if not self.hypothetical(rowid): rowid = -1
            return network.simulate(rowid, query, evidence, N)
        # Static query analysis.
        unwrap = N is None
        if unwrap: N = 1
        exposed = self.outputs[0] in query
        if exposed: query = [q for q in query if q != self.outputs[0]]
        # Marginalize over clusters.
        K = self.crp.clusters[0].gibbs_tables(-1)
        evidences = [merged(evidence, {self.outputs[0]: k}) for k in K]
        lp_evidence_unorm = [network.logpdf(rowid, ev) for ev in evidences]
        Ks = gu.log_pflip(lp_evidence_unorm, array=K, size=N, rng=self.rng)
        counts = {k:n for k, n in enumerate(np.bincount(Ks)) if n > 0}
        evidences = {k: merged(evidence, {self.outputs[0]: k}) for k in counts}
        samples = [network.simulate(rowid, query, evidences[k], counts[k])
            for k in counts]
        # Expose the CRP to the sample.
        if exposed:
            expose = lambda S, k: [merged(l, {self.outputs[0]: k}) for l in S]
            samples = [expose(s, k) for s, k in zip(samples, counts)]
        # Return samples.
        result = list(itertools.chain.from_iterable(samples))
        return result[0] if unwrap else result

    # --------------------------------------------------------------------------
    # Internal simulate/logpdf helpers

    def build_network(self):
        return ImportanceNetwork(
            cgpms=[self.crp.clusters[0]] + self.dims.values(),
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
        if z_b != self.Zr(rowid):
            self.unincorporate(rowid)
            query = merged(
                {d: self.X[d][rowid] for d in self.dims},
                {self.outputs[0]: z_b})
            self.incorporate(rowid, query)
        self._check_partitions()

    def _logpdf_row_gibbs(self, rowid, K):
        return [sum([self._logpdf_cell_gibbs(rowid, dim, k)
            for dim in self.dims.itervalues()]) for k in K]

    def _logpdf_cell_gibbs(self, rowid, dim, k):
        query = {dim.index: self.X[dim.index][rowid]}
        evidence = self._get_evidence(rowid, dim, k)
        # If rowid in cluster k then unincorporate then compute predictive.
        if self.Zr(rowid) == k:
            dim.unincorporate(rowid)
            logp = dim.logpdf(rowid, query, evidence)
            dim.incorporate(rowid, query, evidence)
        else:
            logp = dim.logpdf(rowid, query, evidence)
        return logp

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
        return len(self.X[self.X.keys()[0]])

    def hypothetical(self, rowid):
        return not (0 <= rowid < len(self.Zr()))

    def _populate_evidence(self, rowid, query, evidence):
        """Loads query evidence from the dataset."""
        if evidence is None: evidence = {}
        if self.hypothetical(rowid): return evidence
        data = {c: self.X[c][rowid] for c in self.outputs[1:]
            if c not in evidence and c not in query
            and not isnan(self.X[c][rowid])}
        cluster = {self.outputs[0]: self.Zr(rowid)}
        return merged(evidence, data, cluster)

    def _get_evidence(self, rowid, dim, k):
        """Prepare the evidence for a Dim logpdf/simulate query."""
        inputs = {i: self.X[i][rowid] for i in dim.inputs[1:]}
        cluster = {self.outputs[0]: k}
        return merged(inputs, cluster)

    def _bulk_incorporate(self, dim):
        # XXX Major hack! We should really be creating new Dim objects.
        dim.clusters = {}   # Mapping of cluster k to the object.
        dim.Zr = {}         # Mapping of non-nan rowids to cluster k.
        dim.Zi = {}         # Mapping of nan rowids to cluster k.
        dim.aux_model = dim.create_aux_model()
        for rowid, k in sorted(self.Zr().items(), key=lambda e: e[1]):
            dim.incorporate(
                rowid,
                query={dim.index: self.X[dim.index][rowid]},
                evidence=self._get_evidence(rowid, dim, k))
        assert merged(dim.Zr, dim.Zi) == self.Zr()
        dim.transition_params()

    def _prepare_incorporate(self, cctype):
        distargs = {}
        if cctype_class(cctype).is_conditional():
            if len(self.dims) == 0:
                raise ValueError('Cannot incorporate single conditional dim.')
            distargs['cctypes'] = self._unconditional_cctypes()
            distargs['ccargs'] = self._unconditional_ccargs()
        return distargs

    def _conditional_dims(self):
        """Return conditional dims in sorted order."""
        return filter(lambda d: self.dims[d].is_conditional(),
            sorted(self.dims))

    def _unconditional_dims(self):
        """Return unconditional dims in sorted order."""
        return filter(lambda d: not self.dims[d].is_conditional(),
            sorted(self.dims))

    def _unconditional_cctypes(self):
        dims = [self.dims[i] for i in self._unconditional_dims()]
        return [d.cctype for d in dims]

    def _conditional_cctypes(self):
        dims = [self.dims[i] for i in self._conditional_dims()]
        return [d.cctype for d in dims]

    def _unconditional_ccargs(self):
        dims = [self.dims[i] for i in self._unconditional_dims()]
        return [d.get_distargs() for d in dims]

    def _conditional_ccargs(self):
        dims = [self.dims[i] for i in self._unconditional_dims()]
        return [d.get_distargs() for d in dims]

    # --------------------------------------------------------------------------
    # Data structure invariants.

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha() > 0.
        # Check that the number of dims actually assigned to the view
        # matches the count in Nv.
        rowids = range(self.n_rows())
        Zr, Nk = self.Zr(), self.Nk()
        assert set(Zr.keys()) == set(xrange(self.n_rows()))
        assert set(Zr.values()) == set(Nk)
        for dim in self.dims.itervalues():
            # Assert first output is first input of the Dim.
            assert self.outputs[0] == dim.inputs[0]
            # Ensure number of clusters in each dim in views[v]
            # is the same and as described in the view (K, Nk).
            assignments = merged(dim.Zr, dim.Zi)
            assert assignments == Zr
            assert set(assignments.values()) == set(Nk.keys())
            all_ks = dim.clusters.keys() + dim.Zi.values()
            assert set(all_ks) == set(Nk.keys())
            for k in dim.clusters:
                # Law of conservation of rowids.
                rowids_nan = np.isnan(
                    [self.X[dim.index][r] for r in rowids if Zr[r]==k])
                assert dim.clusters[k].N + np.sum(rowids_nan) == Nk[k]
