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

from math import isinf
from math import isnan

import numpy as np

from scipy.misc import logsumexp

import gpmcc.utils.general as gu

from gpmcc.cgpm import CGpm
from gpmcc.mixtures.dim import Dim
from gpmcc.utils.config import cctype_class
from gpmcc.utils.general import logmeanexp
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
            Global dataset of dimension N x D. The invariant is that
            the data for dim.index should be in X[:,dim.index] and the data
            for rowid should X[rowid,:]. All rows in X will be incorporated.
        dims : list<Dim>
            A list of Dim objects in this View.
        alpha : float, optional
            CRP concentration parameter. If None, selected from grid uniformly
            at random.
        Zr : list<int>, optional
            Starting partiton of rows to categories where Zr[i] is the latent
            clsuter of row i. If None, is sampled from CRP(alpha).
        """
        # Entropy.
        self.rng = gu.gen_rng() if rng is None else rng

        # Inputs
        if inputs:
            raise ValueError('View does not accept inputs.')
        self.inputs = []

        # Dataset.
        self.X = X

        # Outputs
        if outputs:
            assert len(outputs) == len(cctypes) == len(distargs) == len(hypers)
        else:
            outputs = []
        self.outputs = outputs

        # CRP alpha grid.
        self.alpha_grid = gu.log_linspace(1./self.n_rows(), self.n_rows(), 30)
        if alpha is None:
            alpha = self.rng.choice(self.alpha_grid)
        self.alpha = alpha

        # Row partition.
        if Zr is None:
            Zr = gu.simulate_crp(self.n_rows(), alpha, rng=self.rng)
        # Convert Zr to a dictionary.
        self.Zr = {i:z for i,z in enumerate(Zr)}
        # Build Nk dictionary.
        self.Nk = {k:count for k,count in enumerate(np.bincount(Zr)) if count>0}


        # XXX Initialize the CRP CGpm, with an index of 1e7.
        self.crp = Dim(
            [1e7], cctype='crp', hypers={'alpha': alpha}, rng=self.rng)
        self.crp.transition_hyper_grids([1]*self.n_rows())
        if Zr is None:
            for i in xrange(self.n_rows()):
                s = self.crp.simulate(i, [1e7], {-1:0})
                self.crp.incorporate(i, s, {-1:0})
        else:
            for i, z in enumerate(Zr):
                self.crp.incorporate(i, {1e7:z}, {-1:0})
        assert self.Zr == self.crp.clusters[0].data
        assert self.Nk == self.crp.clusters[0].counts
        if not np.allclose(self.crp.hyper_grids['alpha'], self.alpha_grid):
            import ipdb; ipdb.set_trace()

        # Dimensions.
        self.dims = dict()
        for i, c in enumerate(self.outputs):
            dim = Dim(
                outputs=[c], inputs=None, cctype=cctypes[i],
                hypers=hypers[i], distargs=distargs[i], rng=self.rng)
            dim.transition_hyper_grids(self.X[c])
            if dim.is_conditional():
                raise ValueError('Use incorporate for conditional dims.')
            self.incorporate_dim(dim)

        self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, dim, reassign=True):
        """Incorporate the dim into this View. If reassign is False, the row
        partition of dim should match self.Zr already."""
        if reassign:
            distargs = self._prepare_incorporate(dim.cctype)
            dim.distargs.update(distargs)
            self._bulk_incorporate(dim)
        self.dims[dim.index] = dim
        self.outputs = self.dims.keys()
        return dim.logpdf_score()

    def _bulk_incorporate(self, dim):
        # XXX Major hack! We should really be creating new Dim objects.
        dim.clusters = {}   # Mapping of cluster k to the object.
        dim.Zr = {}         # Mapping of non-nan rowids to cluster k.
        dim.Zi = {}         # Mapping of nan rowids to cluster k.
        dim.aux_model = dim.create_aux_model()
        for rowid, k in sorted(self.Zr.items(), key=lambda e: e[1]):
            dim.incorporate(
                rowid,
                query={dim.index: self.X[dim.index][rowid]},
                evidence=self._get_evidence(rowid, dim, k))
        assert merged(dim.Zr, dim.Zi) == self.Zr
        dim.transition_params()

    def _prepare_incorporate(self, cctype):
        distargs = {}
        if cctype_class(cctype).is_conditional():
            if len(self.dims) == 0:
                raise ValueError('Cannot incorporate single conditional dim.')
            distargs['cctypes'] = self._unconditional_cctypes()
            distargs['ccargs'] = self._unconditional_ccargs()
        return distargs

    def unincorporate_dim(self, dim):
        """Remove dim from this View (does not modify)."""
        del self.dims[dim.index]
        self.outputs = self.dims.keys()
        return dim.logpdf_score()

    def incorporate(self, rowid, query, evidence=None):
        """Incorporate an observation into the View.

        Parameters
        ----------
        rowid : int
            Fresh, non-negative rowid.
        query : dict{output:val}
            Keys of the query must exactly be the output (Github issue 89).
            Optionally use {-1:k} for latent cluster assignment of rowid where
            0 <= k <= len(self.Nk). The cluster is a query variable since View
            has a generative model for k, unlike Dim which takes k as evidence.
        """
        k = query.get(-1, 0)
        transition = [rowid] if k is None else []
        if k not in self.Nk:
            self.Nk[k] = 0
        self.Nk[k] += 1
        self.Zr[rowid] = k
        # XXX F ME XXX
        self.crp.incorporate(rowid, {1e7:k}, {-1:0})
        assert self.Zr == self.crp.clusters[0].data
        assert self.Nk == self.crp.clusters[0].counts
        # XXX F ME XXX
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
        k = self.Zr[rowid]
        self.Nk[k] -= 1
        if self.Nk[k] == 0:
            del self.Nk[k]
            for dim in self.dims.itervalues():
                # XXX Abstract in a better way
                del dim.clusters[k]
        del self.Zr[rowid]
        # XXX F ME XXX
        self.crp.unincorporate(rowid)
        assert self.Zr == self.crp.clusters[0].data
        assert self.Nk == self.crp.clusters[0].counts
        # XXX F ME XXX

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
        """Run all the transitions N times."""
        for _ in xrange(N):
            self.transition_rows()
            self.transition_alpha()
            self.transition_column_hypers()

    def transition_alpha(self):
        """Calculate CRP alpha conditionals over grid and transition."""
        logps = [gu.logp_crp_unorm(len(self.Zr), len(self.Nk), alpha)
            for alpha in self.alpha_grid]
        self.alpha = gu.log_pflip(logps, array=self.alpha_grid, rng=self.rng)
        # XXX F ME XXX
        logp_crp_baz = [gu.logp_crp(len(self.Zr), self.Nk_list(), alpha)
            for alpha in self.alpha_grid]
        logp_crp_foo = self.crp._calc_hyper_proposal_logps('alpha')
        assert np.allclose(logp_crp_baz, logp_crp_foo)
        self.crp.clusters[0].alpha = self.alpha
        self.crp.hypers = {'alpha': self.alpha}
        # XXX F ME XXX

    def transition_column_hypers(self, cols=None):
        """Calculate column (dim) hyperparameter conditionals over grid and
        transition."""
        if cols is None:
            cols = self.dims.keys()
        for c in cols:
            self.dims[c].transition_hypers()

    def transition_rows(self, rows=None):
        """Compute row conditions for each cluster and transition."""
        if rows is None:
            rows = self.Zr.keys()
        for rowid in rows:
            self._transition_row(rowid)

    # --------------------------------------------------------------------------
    # logscore.

    def logpdf_score(self):
        """Compute the marginal logpdf CRP assignment and data."""
        logp_crp = gu.logp_crp(len(self.Zr), self.Nk_list(), self.alpha)
        # XXX F ME XXX
        assert np.allclose(logp_crp, self.crp.logpdf_score())
        # XXX F ME XXX
        logp_dims = [dim.logpdf_score() for dim in self.dims.itervalues()]
        return logp_crp + sum(logp_dims)

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence):
        assert isinstance(query, dict)
        assert isinstance(evidence, dict)
        if self._is_hypothetical(rowid):
            return self._logpdf_hypothetical(query, evidence)
        else:
            return self._logpdf_observed(rowid, query, evidence)

    def _logpdf_observed(self, rowid, query, evidence):
        evidence = self._populate_evidence(rowid, query, evidence)
        return self._logpdf_joint(query, evidence, self.Zr[rowid])

    def _logpdf_hypothetical(self, query, evidence):
        # Algorithm. Partition all columns in query and evidence by views.
        # P(xQ|xE) = \sum_z p(xQ|z,xE)p(z|xE)       marginalization
        # Now consider p(z|xE) \propto p(z)p(xE|z)  Bayes rule
        # [term]    [array]
        # p(z)      logp_crp
        # p(xE|z)   logp_evidence
        # p(z|xE)   logp_cluster
        # p(xQ|z)   logp_query
        K = sorted(self.Nk.keys() + [max(self.Nk.keys())+1])
        lp_crp = gu.logp_crp_fresh(len(self.Zr), self.Nk_list(), self.alpha)
        # XXX F ME XXX
        K_prime = self.crp.clusters[0].gibbs_tables(-1)
        lp_crp_2 = [self.crp.logpdf(-1, {1e7: v}, {-1:0}) for v in K]
        assert np.allclose(K, K_prime)
        assert np.allclose(lp_crp_2, lp_crp)
        # XXX F ME XXX
        lp_evidence = [self._logpdf_joint(evidence, {}, k) for k in K]
        if all(isinf(l) for l in lp_evidence): raise ValueError('Inf evidence!')
        lp_cluster = gu.log_normalize(np.add(lp_crp, lp_evidence))
        lp_query = [self._logpdf_joint(query, evidence, k) for k in K]
        return logsumexp(np.add(lp_cluster, lp_query))

    # --------------------------------------------------------------------------
    # simulate

    def simulate(self, rowid, query, evidence, N=None):
        assert isinstance(query, list)
        assert isinstance(evidence, dict)
        if N is None:
            N = 1
        if self._is_hypothetical(rowid):
            return self._simulate_hypothetical(query, evidence, N)
        else:
            return self._simulate_observed(rowid, query, evidence, N)

    def _simulate_observed(self, rowid, query, evidence, N):
        evidence = self._populate_evidence(rowid, query, evidence)
        samples = self._simulate_joint(query, evidence, self.Zr[rowid], N)
        return samples

    def _simulate_hypothetical(self, query, evidence, N, cluster=False):
        """cluster exposes latent cluster of each sample in extra column."""
        K = sorted(self.Nk.keys() + [max(self.Nk.keys())+1])
        lp_crp = gu.logp_crp_fresh(len(self.Zr), self.Nk_list(), self.alpha)
        # XXX F ME XXX
        K_prime = self.crp.clusters[0].gibbs_tables(-1)
        lp_crp_2 = [self.crp.logpdf(-1, {1e7: v}, {-1:0}) for v in K]
        assert np.allclose(K, K_prime)
        assert np.allclose(lp_crp_2, lp_crp)
        # XXX F ME XXX
        lp_evidence = [self._logpdf_joint(evidence, {}, k) for k in K]
        if all(isinf(l) for l in lp_evidence): raise ValueError('Inf evidence!')
        lp_cluster = np.add(lp_crp, lp_evidence)
        ks = gu.log_pflip(lp_cluster, array=K, size=N, rng=self.rng)
        counts = {k:n for k, n in enumerate(np.bincount(ks)) if n > 0}
        samples = [self._simulate_joint(query, evidence, k, counts[k])
            for k in sorted(counts)]
        # XXX HACK! Shoud use a flag in evidence, not kwarg.
        if cluster:
            for s, k in zip(samples, sorted(counts)):
                [l.update({-1: k}) for l in s]
        return list(itertools.chain.from_iterable(samples))

    # --------------------------------------------------------------------------
    # simulate/logpdf helpers

    def no_leafs(self, query, evidence):
        roots = self._unconditional_dims()
        clean_evidence = all(e in roots for e in evidence)
        clean_query = all(q in roots for q in query)
        return clean_evidence and clean_query

    def _simulate_joint(self, query, evidence, k, N):
        assert isinstance(evidence, dict)
        assert isinstance(query, list)
        if self.no_leafs(query, evidence):
            return [self._simulate_unconditional(query, k) for i in xrange(N)]
        # XXX Should we resample ACCURACY times from the prior for 1 sample?
        else:
            ACCURACY = N if self.no_leafs(evidence, {}) else 20*N
            samples, weights = self._weighted_samples(evidence, k, ACCURACY)
            return self._importance_resample(query, samples, weights, N)

    def _logpdf_joint(self, query, evidence, k):
        assert isinstance(evidence, dict)
        assert isinstance(query, dict)
        if self.no_leafs(query, evidence):
            return self._logpdf_unconditional(query, k)
        else:
            ACCURACY = 20
            ev_qr = merged(evidence, query)
            _, weights_eq = self._weighted_samples(ev_qr, k, ACCURACY)
            logp_evidence = 0.
            if evidence:
                _, weights_e = self._weighted_samples(evidence, k, ACCURACY)
                logp_evidence = logmeanexp(weights_e)
            logp_query = logmeanexp(weights_eq) - logp_evidence
            return logp_query

    def _importance_resample(self, query, samples, weights, N):
        indices = gu.log_pflip(weights, size=N, rng=self.rng)
        return [{q: samples[i][q] for q in query} for i in indices]

    def _weighted_samples(self, evidence, k, N):
        # Find roots and leafs indices.
        rts = self._unconditional_dims()
        lfs = self._conditional_dims()
        # Separate root and leaf evidence.
        ev_rts = {e:x for e,x in evidence.iteritems() if e in rts}
        ev_lfs = {e:x for e,x in evidence.iteritems() if e in lfs}
        # Simulate missing roots.
        rts_mis = [r for r in rts if r not in ev_rts]
        rts_sim = [self._simulate_unconditional(rts_mis, k) for i in xrange(N)]
        rts_all = [merged(ev_rts, r) for r in rts_sim]
        # Simulate missing leafs.
        lfs_mis = [l for l in lfs if l not in ev_lfs]
        lfs_sim = [self._simulate_conditional(lfs_mis, r, k) for r in rts_all]
        lfs_all = [merged(ev_lfs, l) for l in lfs_sim]
        # Likelihood of evidence in sample.
        wgt_rts = self._logpdf_unconditional(ev_rts, k)
        wgt_lfs = [self._logpdf_conditional(ev_lfs, r, k) for r in rts_all]
        weights = [wgt_lf + wgt_rts for wgt_lf in wgt_lfs]
        # Combine the entire sample.
        samples = [merged(ra, la) for (ra,la) in zip(rts_all, lfs_all)]
        # Sample and its weight.
        return samples, weights

    def _simulate_unconditional(self, query, k):
        """Simulate query from cluster k, N times."""
        assert not any(self.dims[c].is_conditional() for c in query)
        samples = [self.dims[c].simulate(-1, [c], {-1:k}) for c in query]
        return merged(*samples)

    def _simulate_conditional(self, query, evidence, k):
        """Simulate unconditional query from cluster k."""
        assert set(self._unconditional_dims()) == set(evidence)
        assert all(self.dims[c].is_conditional() for c in query)
        evidence = merged(evidence, {-1:k})
        samples = [self.dims[c].simulate(-1, [c], evidence) for c in query]
        return merged(*samples)

    def _logpdf_unconditional(self, query, k):
        assert not any(self.dims[c].is_conditional() for c in query)
        lps = [self.dims[c].logpdf(-1, {c: query[c]}, {-1: k}) for c in query]
        return sum(lps)

    def _logpdf_conditional(self, query, evidence, k):
        assert all(self.dims[c].is_conditional() for c in query)
        assert set(self._unconditional_dims()) == set(evidence)
        evidence = merged(evidence, {-1: k})
        lps = [self.dims[c].logpdf(-1, {c:query[c]}, evidence) for c in query]
        return sum(lps)

    # --------------------------------------------------------------------------
    # Internal row transition.

    def _transition_row(self, rowid):
        # If singleton then no aux clusters.
        m_aux = [] if self.Nk[self.Zr[rowid]]==1 else [max(self.Nk)+1]
        K = sorted(self.Nk.keys() + m_aux)

        # Probability of row crp assignment to each cluster.
        logp_crp = gu.logp_crp_gibbs(self.Nk, self.Zr, rowid, self.alpha, 1)

        # XXX F ME
        logp_crp_manual = self.crp.clusters[0].gibbs_logps(rowid)
        K_manual = self.crp.clusters[0].gibbs_tables(rowid)
        assert np.allclose(logp_crp, logp_crp_manual)
        assert np.allclose(K, K_manual)
        # XXX F ME


        # Probability of row data in each cluster.
        logp_data = self._logpdf_row_gibbs(rowid, K)

        assert len(logp_data) == len(logp_crp)

        # Sample new cluster.
        p_cluster = np.add(logp_data, logp_crp)
        z_b = gu.log_pflip(p_cluster, array=K, rng=self.rng)

        # Migrate the row.
        if z_b != self.Zr[rowid]:
            self.unincorporate(rowid)
            query = merged(
                {d: self.X[d][rowid] for d in self.dims}, {-1: z_b})
            self.incorporate(rowid, query)
        self._check_partitions()

    def _logpdf_row_gibbs(self, rowid, K):
        return [sum([self._logpdf_cell_gibbs(rowid, dim, k)
            for dim in self.dims.itervalues()]) for k in K]

    def _logpdf_cell_gibbs(self, rowid, dim, k):
        query = {dim.index: self.X[dim.index][rowid]}
        evidence = self._get_evidence(rowid, dim, k)
        # If rowid in cluster k then unincorporate then compute predictive.
        if self.Zr[rowid] == k:
            dim.unincorporate(rowid)
            logp = dim.logpdf(rowid, query, evidence)
            dim.incorporate(rowid, query, evidence)
        else:
            logp = dim.logpdf(rowid, query, evidence)
        return logp

    # --------------------------------------------------------------------------
    # Internal query utils.

    def Nk_list(self):
        return [self.Nk[k] for k in sorted(self.Nk)]

    def n_rows(self):
        return len(self.X[self.X.keys()[0]])

    def _is_hypothetical(self, rowid):
        return not (0 <= rowid < len(self.Zr))

    def _populate_evidence(self, rowid, query, evidence):
        """Builds the evidence for an observed simulate/logpdb query."""
        if self._is_hypothetical(rowid):
            return evidence
        em = [r for r in self.outputs if r not in evidence and r not in query]
        ev = {c: self.X[c][rowid] for c in em if not isnan(self.X[c][rowid])}
        return merged(evidence, ev)

    def _get_evidence(self, rowid, dim, k):
        """Prepare the evidence for a Dim logpdf/simulate query."""
        inputs = {i: self.X[i][rowid] for i in dim.inputs}
        cluster = {-1: k}
        return merged(inputs, cluster)

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
        assert self.alpha > 0.
        # Check that the number of dims actually assigned to the view
        # matches the count in Nv.
        rowids = range(self.n_rows())
        assert set(self.Zr.keys()) == set(xrange(self.n_rows()))
        assert set(self.Zr.values()) == set(self.Nk)
        for dim in self.dims.itervalues():
            # Ensure number of clusters in each dim in views[v]
            # is the same and as described in the view (K, Nk).
            assignments = merged(dim.Zr, dim.Zi)
            assert assignments == self.Zr
            assert set(assignments.values()) == set(self.Nk.keys())
            all_ks = dim.clusters.keys() + dim.Zi.values()
            assert set(all_ks) == set(self.Nk.keys())
            for k in dim.clusters:
                # Law of conservation of rowids.
                rowids_nan = np.isnan(
                    [self.X[dim.index][r] for r in rowids if self.Zr[r]==k])
                assert dim.clusters[k].N + np.sum(rowids_nan) == self.Nk[k]
