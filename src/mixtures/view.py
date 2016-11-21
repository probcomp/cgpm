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
            if not distargs:
                distargs = [None]*len(cctypes)
            if not hypers:
                hypers = [None]*len(cctypes)
            assert len(outputs[1:])==len(cctypes)
            assert len(distargs) == len(cctypes)
            assert len(hypers) == len(cctypes)
        self.outputs = list(outputs)

        # -- Latents -----------------------------------------------------------
        self.exposed_latent = self.outputs[0]

        # -- Row CRP -----------------------------------------------------------
        self.crp = Dim(
            [self.exposed_latent], [-1],
            cctype='crp',
            hypers=None if alpha is None else {'alpha': alpha},
            rng=self.rng)
        self.crp.transition_hyper_grids([1]*self.n_rows())
        if Zr is None:
            for i in xrange(self.n_rows()):
                s = self.crp.simulate(i, [self.exposed_latent], {-1:0})
                self.crp.incorporate(i, s, {-1:0})
        else:
            for i, z in enumerate(Zr):
                self.crp.incorporate(i, {self.exposed_latent: z}, {-1:0})

        # -- Dimensions --------------------------------------------------------
        self.dims = dict()
        for i, c in enumerate(self.outputs[1:]):
            dim = Dim(
                outputs=[c], inputs=[self.exposed_latent], cctype=cctypes[i],
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
        dim.inputs[0] = self.exposed_latent
        if reassign:
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
            Optionally, use {self.exposed_latent: k} for latent cluster assignment
            of rowid. The cluster is a query variable since View
            has a generative model for k, unlike Dim which takes k as evidence.
        """
        n_rows = len(self.X[0])
        if rowid > n_rows:  # if rowid would be skipped for incorporate
            raise ValueError(
                "Rowid cannot be larger than %d" % (n_rows,))
        
        # Cluster assignment
        k = query.get(self.exposed_latent, 0)
        if self.exposed_latent not in query:  # simulate if no cluster 
            k = self.simulate(
                -1, query=[self.exposed_latent], evidence=query).values()[0] 
            # get simulated cluster, conditioned on query

        filled_query = self.fill_in_missing_values(query)
        transition = [rowid] if k is None else []
        self.crp.incorporate(rowid, {self.exposed_latent: k}, {-1: 0})
        for d in self.dims:
            self.dims[d].incorporate(
                rowid,
                query={d: filled_query[d]},
                evidence=self._get_evidence(rowid, self.dims[d], k))
        self.transition_rows(rows=transition)
        self._add_to_dataset(rowid=rowid, query=filled_query)

    def fill_in_missing_values(self, query):
        """
        Fill in missing values in query with NaN.
        """
        exposed_outputs = self.outputs[1:]
        filled_query = {self.exposed_latent: query.get(self.exposed_latent, 0)}
        for d in exposed_outputs:
            filled_query[d] = query.get(d, np.nan)
        return filled_query
    
    def _add_to_dataset(self, rowid, query):
        filled_query = self.fill_in_missing_values(query)
        for c in self.X.keys():
            if rowid < len(self.X[c]):  # if row has been unincorporated
                self.X[c][rowid] = filled_query[c]
            else:  # if row is to be inserted in the end of dataset
                self.X[c] += [filled_query[c]]

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
        self._remove_from_dataset(rowid=rowid)

    def _remove_from_dataset(self, rowid):
        for c in self.X.keys():
            self.X[c][rowid] = np.nan  # puts a placeholder to preserve order 
    # --------------------------------------------------------------------------
    # Update schema.

    def update_cctype(self, col, cctype, distargs=None):
        """Update the distribution type of self.dims[col] to cctype."""
        if distargs is None:
            distargs = {}
        inputs = []
        # XXX Horrid hack.
        if cctype_class(cctype).is_conditional():
            if len(self.dims) == 0:
                raise ValueError('Cannot incorporate single conditional dim.')
            inputs = filter(
                lambda d: d != col and not self.dims[d].is_conditional(),
                sorted(self.dims))
            distargs['inputs'] = {
                'stattypes': [self.dims[i].cctype for i in inputs],
                'statargs': [self.dims[i].get_distargs() for i in inputs]
            }
        D_old = self.dims[col]
        D_new = Dim(
            outputs=[col], inputs=[self.exposed_latent]+inputs,
            cctype=cctype, distargs=distargs, rng=self.rng)
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

    def transition_dim_grids(self, cols=None):
        if cols is None:
            cols = self.dims.keys()
        for c in cols:
            self.dims[c].transition_hyper_grids(self.X[c])

    def transition_rows(self, rows=None):
        if rows is None:
            rows = self.Zr().keys()
        rows = self.rng.permutation(rows)
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
        # Now consider p(z|xE) \propto p(z,xE)      Bayes rule
        # p(z,xE)                                   logp_evidence_unorm
        # p(z|xE)                                   logp_evidence
        # p(xQ|z,xE)                                logp_query
        evidence = self._populate_evidence(rowid, query, evidence)
        network = self.build_network()
        # Condition on cluster.
        if self.exposed_latent in evidence:
            # XXX DETERMINE ME!
            if not self.hypothetical(rowid): rowid = -1
            return network.logpdf(rowid, query, evidence)
        elif self.exposed_latent in query:
            # No need to marginalize P(xQ, z)
            return network.logpdf(rowid, query, evidence)
        # Marginalize over clusters.
        K = self.crp.clusters[0].gibbs_tables(-1)
        evidences = [merged(evidence, {self.exposed_latent: k}) for k in K]
        lp_evidence_unorm = [network.logpdf(rowid, ev) for ev in evidences]
        lp_evidence = gu.log_normalize(lp_evidence_unorm)
        lp_query = [network.logpdf(rowid, query, ev) for ev in evidences]
        return gu.logsumexp(np.add(lp_evidence, lp_query))

    # --------------------------------------------------------------------------
    # logpdf multirow

    def logpdf_multirow(self, query, evidence=None):
        # store query and evidence rows already in the dataset 
        # unincorporate those query and evidence rows

        # compute the joint of query and evidence (given categories)
        # compute the joint of evidence (given categories)

        # reincorporate rows
        return np.nan

    def _joint_logpdf_multirow(self, query, evidence):
        """
        query - {rowid: {outputs: value, exposed_latent: value}}
        evidence = {rowid: {exposed_latent: value}}
        """
        # TODO:
        # [ ] check that all the rows in evidence are in query
        # [ ] check that no latent column is both in query and clusters
        # [ ] check that no observable is in evidence

        # Store query rows already in dataset
        # rowids_in_input = set(query.keys() + evidence.keys())
        # for row in rowids_in_input:  # For rows in query and evidence
        #     if not self.hypothetical(rowid):  # if row in dataset
        #     T = [rowid: {}]  # Store values in T
        # For rows in T
        # Unincorporate rows from the GPM

        # Compute Joint Logpdf applying the chain rule recursively

        # for rows in T
        # reincorporate rows to the dataset
        return np.nan

    def _joint_logpdf_multirow_helper(self, counter, query, evidence):
        p = - np.float("inf")  # initialize output as log space zero
        rowid = query.keys()[counter]  # retrieve id of current row 

        if counter == len(query):  # base case, end of chain rule
            p = 0  

        elif rowid in evidence.keys():  # if current row has an assigned cluster 
            p_row = self.logpdf( 
                rowid=rowid, query=query[rowid], evidence=evidence[rowid]
            )  # evaluate log p(query | cluster)
            self.incorporate(
                rowid=rowid, query=merged(query[rowid], evidence[rowid])
            )  # incorporate row with cluster given by evidence
            p_row += self._joint_logpdf_multirow_helper(
                counter+1, query, evidence
            )  # recursion: chain rule p(row)*p(other_rows|row)
            p = gu.logsumexp(p, p_row)  # marginalize out clusters
            self.unincorporate(rowid=rowid)  # unincorporate incorporated row

        else:  # if current row is not in evidence
            K = self.crp.clusters[0].gibbs_tables(-1)  # get possible clusters
            for k in K:  # for each possible cluster assignments
                p_row = self.logpdf(
                    rowid=rowid, query=merged(query[rowid], evidence[rowid])
                )  # compute the single row joint 
                self.incorporate(
                    rowid=rowid, query=merged(query[rowid], evidence[rowid])
                )  # incorporate row into table into respective cluster
                p_row += self._joint_logpdf_multirow_helper(
                    counter+1, query, evidence
                )  # recursion: chain rule p(row)*p(other_rows|row)
                p = gu.logsumexp(p, p_row)  # marginalize out clusters
                self.unincorporate(rowid=rowid)  # unincorporate current row

        return p      # return output probability

    # --------------------------------------------------------------------------
    # simulate

    def simulate(self, rowid, query, evidence=None, N=None):
        evidence = self._populate_evidence(rowid, query, evidence)
        network = self.build_network()
        # Condition on cluster.
        if self.exposed_latent in evidence:
            # XXX DETERMINE ME!
            if not self.hypothetical(rowid): rowid = -1
            return network.simulate(rowid, query, evidence, N)
        # Static query analysis.
        unwrap = N is None
        if unwrap: N = 1
        exposed = self.exposed_latent in query
        if exposed: query = [q for q in query if q != self.exposed_latent]
        K = [0]
        # If there are multiple clusters, marginalize over them
        if self.crp.clusters[0].counts:
            K = self.crp.clusters[0].gibbs_tables(-1)
        evidences = [merged(evidence, {self.exposed_latent: k}) for k in K]
        lp_evidence_unorm = [network.logpdf(rowid, ev) for ev in evidences]
        Ks = gu.log_pflip(lp_evidence_unorm, array=K, size=N, rng=self.rng)
        counts = {k:n for k, n in enumerate(np.bincount(Ks)) if n > 0}
        evidences = {k: merged(evidence, {self.exposed_latent: k}) for k in counts}
        samples = [network.simulate(rowid, query, evidences[k], counts[k])
            for k in counts]
        # Expose the CRP to the sample.
        if exposed:
            expose = lambda S, k: [merged(l, {self.exposed_latent: k}) for l in S]
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
                {self.exposed_latent: z_b})
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
        return rowid not in self.Zr() 

    def _populate_evidence(self, rowid, query, evidence):
        """Loads query evidence from the dataset."""
        if evidence is None: evidence = {}
        if self.hypothetical(rowid): return evidence
        data = {c: self.X[c][rowid] for c in self.outputs[1:]
            if c not in evidence and c not in query
            and not isnan(self.X[c][rowid])}
        cluster = {self.exposed_latent: self.Zr(rowid)}
        return merged(evidence, data, cluster)

    def _get_evidence(self, rowid, dim, k):
        """Prepare the evidence for a Dim logpdf/simulate query."""
        inputs = {i: self.X[i][rowid] for i in dim.inputs[1:]}
        cluster = {self.exposed_latent: k}
        return merged(inputs, cluster)

    def _bulk_incorporate(self, dim):
        # XXX Major hack! We should really be creating new Dim objects.
        dim.clusters = {}   # Mapping of cluster k to the object.
        dim.Zr = {}         # Mapping of non-nan rowids to cluster k.
        dim.Zi = {}         # Mapping of nan rowids to cluster k.
        dim.aux_model = dim.create_aux_model()
        for rowid, k in self.Zr().iteritems():
            dim.incorporate(
                rowid,
                query={dim.index: self.X[dim.index][rowid]},
                evidence=self._get_evidence(rowid, dim, k))
        assert merged(dim.Zr, dim.Zi) == self.Zr()
        dim.transition_params()

    # --------------------------------------------------------------------------
    # Data structure invariants.

    def _check_partitions(self):
        if not cu.check_env_debug():
            return
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
            assert self.exposed_latent == dim.inputs[0]
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
        for c in self.outputs[1:]:
            metadata['cctypes'].append(self.dims[c].cctype)
            metadata['hypers'].append(self.dims[c].hypers)
            metadata['distargs'].append(self.dims[c].distargs)

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
