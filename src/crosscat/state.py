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

import cPickle as pickle
import copy
import itertools
import sys
import time

from math import isnan

import matplotlib.pyplot as plt
import numpy as np

import gpmcc.utils.general as gu
import gpmcc.utils.plots as pu
import gpmcc.utils.validation as vu

from gpmcc.cgpm import CGpm
from gpmcc.mixtures.dim import Dim
from gpmcc.mixtures.view import View
from gpmcc.utils.general import logmeanexp


class State(CGpm):
    """CGpm representing Crosscat, built as a composition of smaller CGpms."""

    def __init__(self, X, outputs=None, inputs=None, cctypes=None,
            distargs=None, Zv=None, Zrv=None, alpha=None, view_alphas=None,
            hypers=None, Cd=None, Ci=None, Rd=None, Ri=None, iterations=None,
            rng=None):
        """Construct State GPM with initial conditions and constraints."""
        # Seed.
        self.rng = gu.gen_rng() if rng is None else rng

        # Inputs
        if inputs:
            raise ValueError('State does not accept inputs.')
        self.inputs = []

        # Dataset.
        X = np.asarray(X)
        if not outputs:
            outputs = range(X.shape[1])
        else:
            assert len(outputs) == X.shape[1]
            assert all(o >= 0 for o in outputs)
        self.outputs = outputs
        self.X = {c: X[:,i].tolist() for i,c in enumerate(self.outputs)}

        # State CRP alpha grid.
        self.alpha_grid = gu.log_linspace(1./self.n_cols(), self.n_cols(), 30)

        # State CRP alpha.
        if alpha is None:
            self.alpha = self.rng.choice(self.alpha_grid)
        else:
            self.alpha = alpha

        # View row partitions.
        if Zrv is None:
            Zrv = [None] * len(self.outputs)
        else:
            assert len(Zrv) == len(set(Zv))

        # Constraints.
        self.Cd = [] if Cd is None else Cd
        self.Ci = [] if Ci is None else Ci
        self.Rd = {} if Rd is None else Rd
        self.Ri = {} if Ri is None else Ri
        # XXX Github issue #13.
        if len(self.Cd) > 0:
            raise ValueError('Dependency constraints not yet implemented.')

        # Column partition.
        if Zv is None:
            if self.Cd or self.Ci:
                Zv = gu.simulate_crp_constrained(
                    self.n_cols(), self.alpha, self.Cd, self.Ci, self.Rd,
                    self.Ri, rng=self.rng)
            else:
                Zv = gu.simulate_crp(self.n_cols(), self.alpha, rng=self.rng)
        # Convert Zv to a dictionary.
        assert len(Zv) == len(self.outputs)
        self.Zv = {i:z for i, z in zip(self.outputs, Zv)}


        # View data.
        if cctypes is None: cctypes = [None] * len(self.outputs)
        if distargs is None: distargs = [None] * len(self.outputs)
        if hypers is None: hypers = [None] * len(self.outputs)
        if view_alphas is None: view_alphas = [None] * len(self.outputs)

        # Views.
        self.views = []
        for v in sorted(set(self.Zv.values())):
            v_outputs = [o for o in self.outputs if Zv[o] == v]
            v_cctypes = [cctypes[c] for c in v_outputs]
            v_distargs = [distargs[c] for c in v_outputs]
            v_hypers = [hypers[c] for c in v_outputs]
            view = View(
                self.X, outputs=v_outputs, inputs=None, Zr=Zrv[v],
                alpha=view_alphas[v], cctypes=v_cctypes, distargs=v_distargs,
                hypers=v_hypers, rng=self.rng)
            self.views.append(view)

        # Iteration metadata.
        self.iterations = iterations if iterations is not None else {}

        # Predictors and their parents.
        self.counter = itertools.count(start=1)
        self.accuracy = 50
        self.predictors = {}
        self.parents = {}

        # Validate.
        self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, T, outputs, inputs=None, cctype=None,
            distargs=None, v=None):
        """Incorporate a new Dim into this State with data T."""
        if len(T) != self.n_rows():
            raise ValueError('%d rows required: %d' % (self.n_rows(), len(T)))
        if len(outputs) != 1:
            raise ValueError('Univariate outputs only: %s.' % outputs)
        if outputs[0] in self.outputs:
            raise ValueError('outputs exist: %s, %s.' % (outputs, self.outputs))
        if inputs:
            raise ValueError('inputs unsupported: %s.' % inputs)
        # Append to outputs.
        col = outputs[0]
        self.X[col] = T
        self.outputs.append(col)
        # XXX Does not handle conditional models; consider moving to view?
        D = Dim(
            outputs=outputs, inputs=inputs, cctype=cctype,
            distargs=distargs, rng=self.rng)
        D.transition_hyper_grids(self.X[col])
        # If v unspecified then transition the col.
        transition = [col] if v is None else []
        # Incorporate dim into view.
        v = 0 if v is None else v
        if 0 <= v < self.n_views():
            view = self.views[v]
        elif v == self.n_views():
            view = View(self.X, rng=self.rng)
            self._append_view(view)
        view.incorporate_dim(D)
        self.Zv[col] = v
        # Transition.
        self.transition_columns(cols=transition)
        self.transition_column_hypers(cols=[col])
        # Validate.
        self._check_partitions()

    def unincorporate_dim(self, col):
        """Unincorporate the Dim whose output is col."""
        if self.n_cols() == 1:
            raise ValueError('State has only one dim, cannot unincorporate.')
        if col not in self.outputs:
            raise ValueError('col does not exist: %s, %s.')
        # Find the dim and its view.
        d_del = self.dim_for(col)
        v_del = self.Zv[col]
        self.views[v_del].unincorporate_dim(d_del)
        # Clear a singleton.
        if self.Nv(v_del) == 0:
            self._delete_view(v_del)
        # Clear data, outputs, and view assignment.
        del self.X[col]
        del self.outputs[self.outputs.index(col)]
        del self.Zv[col]
        # Validate.
        self._check_partitions()

    def incorporate(self, rowid, query, evidence=None):
        # Validation.
        if not self._is_hypothetical(rowid): # XXX Only allow new rows for now.
            raise ValueError('Cannot incorporate non-hypothetical: %d' % rowid)
        if not set.issubset(set(q for q in query if q>=0), set(self.outputs)):
            raise ValueError(
                'Query must be subset of outputs: %s, %s.'
                % (query, self.outputs))
        if any(not 0 <= -(q+1) < len(self.views) for q in query if q < 0):
            raise ValueError('Invalid view: %s.' % query)
        if any(isnan(v) for v in query.values()):
            raise ValueError('Cannot incorporate nan: %s.' % query)
        if evidence is None:
            evidence = {}
        # Append the observation to dataset.
        def update_list(c):
            if self._is_hypothetical(rowid):
                self.X[c].append(query.get(c, float('nan')))
            elif c in query:
                assert isnan(self.X[c][rowid])
                self.X[c][rowid] = query[c]
            return self.X[c]
        self.X = {c: update_list(c) for c in self.outputs}
        # Tell the views.
        if self._is_hypothetical(rowid):
            rowid = self.n_rows()-1
        for v, view in enumerate(self.views):
            qv = {d: self.X[d][rowid] for d in view.dims}
            kv = {-1: query[-(v+1)]} if -(v+1) in query else {}
            view.incorporate(rowid, gu.merged(qv, kv))
        # Validate.
        self._check_partitions()

    def unincorporate(self, rowid):
        raise NotImplementedError('Functionality disabled, Github issue #83.')

    # --------------------------------------------------------------------------
    # Schema updates.

    def update_cctype(self, col, cctype, distargs=None):
        """Update the distribution type of self.dims[col] to cctype."""
        assert col in self.outputs
        self.view_for(col).update_cctype(
            col, cctype, distargs=distargs)
        self.transition_column_hyper_grids(cols=[col])
        self.transition_column_params(cols=[col])
        self.transition_column_hypers(cols=[col])
        self._check_partitions()

    def update_foreign_predictor(self, predictor, parents):
        """XXX REWRITE ME!."""
        # Foreign predictors indexed from -1, -2, ... no cycles.
        index = -next(self.counter)
        if any(p in self.predictors for p in parents):
            raise ValueError('No chained predictors.')
        self.predictors[index] = predictor
        self.parents[index] = parents
        return index

    def remove_foreign_predictor(self, index):
        """XXX REWRITE ME!."""
        if index not in self.predictors:
            raise ValueError('Predictor %s never hooked.' % str(index))
        del self.predictors[index]

    # --------------------------------------------------------------------------
    # logscore.

    def logpdf_score(self):
        """Compute the joint density of latents and data p(theta,Z,X|CC)."""
        return gu.logp_crp(len(self.Zv), self.Nv(), self.alpha) + \
            sum(v.logpdf_score() for v in self.views)

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence=None):
        """Compute density of query under posterior predictive distirbution."""
        if evidence is None:
            evidence = {}
        assert isinstance(query, dict)
        assert isinstance(evidence, dict)
        vu.validate_query_evidence(
            self.X, rowid, self._is_hypothetical(rowid),
            query, evidence=evidence)
        evidence = self._populate_evidence(rowid, query, evidence)
        return self._logpdf_joint(rowid, query, evidence)

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, query, evidence=None, N=None):
        """Simulate from the posterior predictive distirbution."""
        if evidence is None:
            evidence = {}
        if N is None:
            N = 1
        assert isinstance(query, list)
        assert isinstance(evidence, dict)
        vu.validate_query_evidence(
            self.X, rowid, self._is_hypothetical(rowid),
            query, evidence=evidence)
        evidence = self._populate_evidence(rowid, query, evidence)
        return self._simulate_joint(rowid, query, evidence, N)

    # --------------------------------------------------------------------------
    # simulate/logpdf helpers

    def _populate_evidence(self, rowid, query, evidence):
        """Builds the evidence for an observed simulate/logpdb query."""
        if self._is_hypothetical(rowid):
            return evidence
        em = [r for r in self.outputs if r not in evidence and r not in query]
        ev = {c: self.X[c][rowid] for c in em if not isnan(self.X[c][rowid])}
        return gu.merged(evidence, ev)

    def _no_leafs(self, query, evidence):
        return True
        # if query and isinstance(query[0], tuple): query = [q[0] for q in query]
        # clean_evidence = all(e[0] >= 0 for e in evidence)
        # clean_query = all(q >= 0 for q in query)
        # return clean_evidence and clean_query

    def _simulate_joint(self, rowid, query, evidence, N):
        if self._no_leafs(query, evidence):
            return self._simulate_roots(rowid, query, evidence, N)
        # XXX Should we resample ACCURACY times from the prior for 1 sample?
        ACC = N if self._no_leafs(evidence, []) else self.accuracy*N
        samples, weights = self._weighted_samples(rowid, query, evidence, ACC)
        return self._importance_resample(samples, weights, N)

    def _logpdf_joint(self, rowid, query, evidence):
        if self._no_leafs(query, evidence):
            return self._logpdf_roots(rowid, query, evidence)
        ACC = self.accuracy
        _, w_joint = self._weighted_samples(rowid, [], evidence+query, ACC)
        logp_evidence = 0.
        if evidence:
            _, w_marg = self._weighted_samples(rowid, [], evidence, ACC)
            logp_evidence = logmeanexp(w_marg)
        logp_query = logmeanexp(w_joint) - logp_evidence
        return logp_query

    def _importance_resample(self, samples, weights, N):
        indices = gu.log_pflip(weights, size=N, rng=self.rng)
        return [samples[i] for i in indices]

    def _weighted_samples(self, rowid, query, evidence, N):
        """Optmized to simulate only required nodes."""
        ev = sorted(evidence)
        # Find roots and leafs indices.
        rts = range(self.n_cols())
        lfs = sorted(self.predictors.keys())
        # Separate root and leaf evidence.
        rts_ev = [e for e in ev if e[0] in rts]
        lfs_ev = [e for e in ev if e[0] in lfs]
        # Separate root and leaf query.
        lfs_obs = [e[0] for e in lfs_ev]
        lfs_qry = [l for l in query if l in lfs]
        # Find obs, qry, and aux.
        rts_obs = [e[0] for e in rts_ev]
        rts_qry = [q for q in query if q in rts]
        rts_aux = self._aux_roots(rts_obs, lfs_obs, rts_qry, lfs_qry)
        # Simulate all required roots.
        rts_mis = rts_qry + rts_aux
        rts_sim_aux = self._simulate_roots(rowid, rts_mis, rts_ev, N)\
            if rts_mis else []
        rts_all = [rts_ev + zip(rts_mis, sample) for sample in rts_sim_aux]\
            if rts_mis else [rts_ev]*N
        # Extract query roots.
        rts_sim = [sample[:len(rts_qry)] for sample in rts_sim_aux]
        # Simulate queried leafs.
        lfs_sim = [self._simulate_leafs(rowid, lfs_qry, r) for r in rts_all]
        if rts_sim: assert len(rts_sim) == len(lfs_sim)
        rts_dr = [{q:draw[i] for i,q in enumerate(rts_qry)} for draw in rts_sim]
        lfs_dr = [{q:draw[i] for i,q in enumerate(lfs_qry)} for draw in lfs_sim]
        for r,l in zip(rts_dr, lfs_dr): r.update(l)
        samples = [[s[q] for q in query] for s in rts_dr]
        # Sample and its weight.
        weights = [self._logpdf_roots(rowid, rts_ev, []) +
            self._logpdf_leafs(rowid, lfs_ev, r) for r in rts_all]
        return samples, weights

    def _aux_roots(self, rts_obs, lfs_obs, rts_qry, lfs_qry):
        assert not any(set(rts_obs)&set(lfs_obs)&set(rts_qry)&set(lfs_qry))
        required = set([v for p in lfs_qry+lfs_obs for v in self.parents[p]])
        rts_seen = rts_obs + rts_qry
        return [r for r in required if r not in rts_seen]

    def _simulate_roots(self, rowid, query, evidence, N):
        assert all(c not in self.predictors for c in query)
        queries, evidences = vu.partition_query_evidence(
            self.Zv, query, evidence)
        samples = [self.views[v].simulate(rowid, queries[v],
            evidence=evidences.get(v, {}), N=N) for v in queries]
        return [gu.merged(*s) for s in zip(*samples)]

    def _simulate_leafs(self, rowid, query, evidence):
        assert all(c in self.predictors for c in query)
        ev_lookup = dict(evidence)
        ev_set = set(ev_lookup.keys())
        all(set.issubset(set(self.parents[c]), ev_set) for c in query)
        ys = [[ev_lookup[p] for p in self.parents[c]] for c in query]
        return [self.predictors[c].simulate(rowid, y) for c,y in zip(query, ys)]

    def _logpdf_roots(self, rowid, query, evidence):
        assert all(c not in self.predictors for c in query)
        queries, evidences = vu.partition_query_evidence(
            self.Zv, query, evidence)
        return sum([self.views[v].logpdf(rowid, queries[v], evidences.get(v,{}))
            for v in queries])

    def _logpdf_leafs(self, rowid, query, evidence):
        assert all(c in self.predictors for c, x in query)
        ev_lookup = dict(evidence)
        ev_set = set(ev_lookup.keys())
        all(set.issubset(set(self.parents[c]), ev_set) for c, x in query)
        ys = [[ev_lookup[p] for p in self.parents[c]] for c, x in query]
        return sum([self.predictors[c].logpdf(rowid, x, y)
            for (c,x), y in zip(query, ys)])

    # --------------------------------------------------------------------------
    # Bulk operations

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None):
        """Evaluate multiple queries at once, used by Engine."""
        if evidences is None: evidences = [{} for i in xrange(len(rowids))]
        if Ns is None: Ns = [1 for i in xrange(len(rowids))]
        assert len(rowids) == len(queries) == len(evidences) == len(Ns)
        return [self.simulate(r, q, e, n)
            for (r, q, e, n) in zip(rowids, queries, evidences, Ns)]

    def logpdf_bulk(self, rowids, queries, evidences=None):
        """Evaluate multiple queries at once, used by Engine."""
        if evidences is None: evidences = [{} for _ in xrange(len(rowids))]
        assert len(rowids) == len(queries) == len(evidences)
        return [self.logpdf(r, q, e)
            for (r, q, e) in zip(rowids, queries, evidences)]

    # --------------------------------------------------------------------------
    # Mutual information

    def mutual_information(self, col0, col1, evidence=None, N=None):
        """Computes the mutual information MI(col0:col1|evidence)."""
        if N is None: N = 1000
        if evidence is None: evidence = {}
        def samples_logpdf(samples, evidence):
            assert len(samples) == N
            return self.logpdf_bulk([-1]*N, samples, [evidence]*N)
        # MI or entropy?
        if col0 != col1:
            samples = self.simulate(-1, [col0, col1], evidence=evidence, N=N)
            PXY = samples_logpdf(samples, evidence)
            PX = samples_logpdf([{col0: s[col0]} for s in samples], evidence)
            PY = samples_logpdf([{col1: s[col1]} for s in samples], evidence)
            return (np.sum(PXY) - np.sum(PX) - np.sum(PY)) / N
        else:
            samples = self.simulate(-1, [col0], evidence=evidence, N=N)
            PX = samples_logpdf([{col0: s[col0]} for s in samples], evidence)
            return - np.sum(PX) / N

    def conditional_mutual_information(self, col0, col1, evidence, T=None,
            N=None):
        """Computes conditional mutual information MI(col0:col1|evidence)."""
        if T is None: T = 100
        samples = self.simulate(-1, evidence, N=T)
        mi = sum(self.mutual_information(
            col0, col1, evidence=zip(evidence, s), N=N) for s in samples)
        return mi / T

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N=None, S=None, kernels=None, target_rows=None,
            target_cols=None, target_views=None, do_plot=False,
            do_progress=True):
        """Run targeted inference kernels."""
        if N is None and S is None:
            N = 1

        # Default order of kernel is important.
        _kernel_functions = [
            ('alpha',
                lambda : self.transition_alpha()),
            ('view_alphas',
                lambda : self.transition_view_alphas(views=target_views)),
            ('column_params',
                lambda : self.transition_column_params(cols=target_cols)),
            ('column_hypers',
                lambda : self.transition_column_hypers(cols=target_cols)),
            ('rows',
                lambda : self.transition_rows(
                    views=target_views, rows=target_rows)),
            ('columns' ,
                lambda : self.transition_columns(cols=target_cols))
        ]

        _kernel_lookup = dict(_kernel_functions)
        if kernels is None:
            kernels = [k[0] for k in _kernel_functions]

        def _proportion_done(N, S, iters, start):
            if S is None:
                p_seconds = 0
            else:
                p_seconds = (time.time() - start) / S
            if N is None:
                p_iters = 0
            else:
                p_iters = float(iters)/N
            return max(p_iters, p_seconds)

        if do_plot:
            fig, layout = self.plot()

        iters = 0
        start = time.time()
        while True:
            for k in kernels:
                p = _proportion_done(N, S, iters, start)
                if p >= 1.:
                    if do_progress:
                        self._do_progress(p)
                    break
                _kernel_lookup[k]()
                self.iterations[k] = self.iterations.get(k,0) + 1
                if do_progress:
                    self._do_progress(p)
                if do_plot:
                    self._do_plot(fig, layout)
                    plt.pause(1e-4)
            else:
                iters += 1
                continue
            break
        if do_progress:
            print 'Completed: %d iterations in %f seconds.' % \
                (iters, time.time()-start)

    def transition_alpha(self):
        """Transition CRP concentration of State."""
        logps = [gu.logp_crp_unorm(self.n_cols(), self.n_views(), alpha)
            for alpha in self.alpha_grid]
        index = gu.log_pflip(logps, rng=self.rng)
        self.alpha = self.alpha_grid[index]

    def transition_view_alphas(self, views=None):
        """Transition CRP concentration of the Views."""
        if views is None:
            views = xrange(self.n_views())
        for v in views:
            self.views[v].transition_alpha()

    def transition_column_params(self, cols=None):
        """Transition uncollapsed Dim parmaters."""
        if cols is None:
            cols = self.outputs
        for c in cols:
            self.dim_for(c).transition_params()

    def transition_column_hypers(self, cols=None):
        """Transition Dim hyperparmaters."""
        if cols is None:
            cols = self.outputs
        for c in cols:
            self.dim_for(c).transition_hypers()

    def transition_column_hyper_grids(self, cols=None):
        """Transition Dim hyperparameter grids."""
        if cols is None:
            cols = self.outputs
        for c in cols:
            self.dim_for(c).transition_hyper_grids(self.X[c])

    def transition_rows(self, views=None, rows=None):
        """Transition row CRP assignments in Views."""
        if self.n_rows() == 1:
            return
        if views is None:
            views = xrange(self.n_views())
        for v in views:
            self.views[v].transition_rows(rows=rows)

    def transition_columns(self, cols=None, m=2):
        """Transition Dim CRP assignments in State."""
        if self.n_cols() == 1:
            return
        if cols is None:
            cols = self.outputs
        cols = self.rng.permutation(cols)
        for c in cols:
            self._transition_column(c, m)

    # --------------------------------------------------------------------------
    # Helpers

    def n_rows(self):
        """Number of incorporated rows."""
        return len(self.X[self.outputs[0]])

    def n_cols(self):
        """Number of incorporated columns."""
        return len(self.outputs)

    def cctypes(self):
        """DistributionGpm name of each Dim."""
        return [d.get_name() for d in self.dims()]

    def distargs(self):
        """DistributionGpm distargs of each Dim."""
        return [d.get_distargs() for d in self.dims()]

    # --------------------------------------------------------------------------
    # Plotting

    def plot(self):
        """Plots observation histogram and posterior distirbution of Dims."""
        layout = pu.get_state_plot_layout(self.n_cols())
        fig = plt.figure(
            num=None,
            figsize=(layout['plot_inches_y'], layout['plot_inches_x']), dpi=75,
            facecolor='w', edgecolor='k', frameon=False, tight_layout=True)
        self._do_plot(fig, layout)
        plt.ion(); plt.show()
        return fig, layout

    def _do_plot(self, fig, layout):
        # Do not plot more than 6 by 4.
        if self.n_cols() > 24:
            return
        fig.clear()
        for dim in self.dims():
            index = dim.index
            ax = fig.add_subplot(layout['plots_x'], layout['plots_y'], index+1)
            dim.plot_dist(self.X[dim.index], ax=ax)
            ax.text(
                1,1, "K: %i " % len(dim.clusters),
                transform=ax.transAxes, fontsize=12, weight='bold',
                color='blue', horizontalalignment='right',
                verticalalignment='top')
            ax.grid()
        plt.draw()

    def _do_progress(self, percentage):
        progress = ' ' * 30
        fill = int(percentage * len(progress))
        progress = '[' + '=' * fill + progress[fill:] + ']'
        print '{} {:1.2f}%\r'.format(progress, 100 * percentage),
        sys.stdout.flush()

    # --------------------------------------------------------------------------
    # Accessors

    def dim_for(self, c):
        """Dim for column c."""
        return self.view_for(c).dims[c]

    def dims(self):
        """All Dims."""
        return [self.view_for(c).dims[c] for c in self.outputs]

    def view_for(self, c):
        """View from Dim c."""
        return self.views[self.Zv[c]]

    def n_views(self):
        """Number of Views."""
        return len(self.views)

    def Nv(self, v=None):
        """Number of Dims in View v."""
        if v is not None:
            return len(self.views[v].dims)
        return [len(view.dims) for view in self.views]

    # --------------------------------------------------------------------------
    # Inference helpers.

    def _transition_column(self, col, m):
        """Gibbs on col assignment to Views, with m auxiliary parameters"""
        # XXX Disable col transitions if \exists conditional model anywhere.
        if any(d.is_conditional() for d in self.dims()):
            raise ValueError('Cannot transition columns with conditional dims.')

        # Some reusable variables.
        v_a = self.Zv[col]

        def is_member(view, dim):
            return view is not None and dim.index in view.dims

        # Compute probability of dim data under view partition.
        def get_data_logp(view, dim):
            if is_member(view, dim):
                return get_data_logp_current(view, dim)
            else:
                return get_data_logp_other(view, dim)

        def get_data_logp_current(view, dim):
            logp = view.unincorporate_dim(dim)
            view.incorporate_dim(dim, reassign=dim.is_collapsed())
            return logp

        def get_data_logp_other(view, dim):
            logp = view.incorporate_dim(dim)
            view.unincorporate_dim(dim)
            return logp

        # Reuse collapsed, deepcopy uncollapsed.
        def get_prop_dim(view, dim):
            if dim.is_collapsed() or is_member(view, dim):
                return dim
            else:
                return copy.deepcopy(dim)

        # Existing views.
        dprops = [get_prop_dim(view, self.dim_for(col)) for view in self.views]
        logp_data = [get_data_logp(view, dim) for (view, dim)
            in zip(self.views, dprops)]

        # Auxiliary views.
        m_aux = range(m-1) if self.Nv(self.Zv[col]) == 1 else range(m)
        dprops_aux = [get_prop_dim(None, self.dim_for(col)) for _ in m_aux]
        vprops_aux = [View(self.X, rng=self.rng) for _ in m_aux]

        logp_data_aux = [get_data_logp(view, dim)
            for (view, dim) in zip(vprops_aux, dprops_aux)]

        # Extend data structs with auxiliary proposals.
        dprops.extend(dprops_aux)
        logp_data.extend(logp_data_aux)

        # Compute the CRP probabilities.
        logp_crp = gu.logp_crp_gibbs(self.Nv(), self.Zv, col, self.alpha, m)
        assert len(logp_data) == len(logp_crp)

        # Overall view probabilities.
        p_view = np.add(logp_data, logp_crp)

        # Enforce independence constraints.
        avoid = [a for p in self.Ci if col in p for a in p if a != col]
        for a in avoid:
            p_view[self.Zv[a]] = float('-inf')

        # Draw view.
        v_b = gu.log_pflip(p_view, rng=self.rng)
        D = dprops[v_b]

        if v_a != v_b:
            self.views[v_a].unincorporate_dim(D)
            if v_b >= self.n_views():
                v_b = self._append_view(vprops_aux[v_b-self.n_views()])
            self.views[v_b].incorporate_dim(D, reassign=D.is_collapsed())
            # Accounting
            self.Zv[col] = v_b
            # Delete empty view?
            if self.Nv(v_a) == 0:
                self._delete_view(v_a)
        else:
            self.views[v_a].incorporate_dim(D, reassign=D.is_collapsed())

        self._check_partitions()

    def _delete_view(self, v):
        assert self.Nv(v) == 0
        del self.views[v]
        adjust = lambda i: i-1 if v < i else i
        self.Zv = {c: adjust(self.Zv[c]) for c in self.outputs}

    def _append_view(self, view):
        """Append a view and return and its index."""
        assert len(view.dims) == 0
        self.views.append(view)
        return self.n_views()-1

    def _is_hypothetical(self, rowid):
        return not 0 <= rowid < self.n_rows()

    # --------------------------------------------------------------------------
    # Data structure invariants.

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha > 0.
        # All outputs should be in the dataset keys.
        assert all([c in self.X.keys() for c in self.outputs])
        # Zv and dims should match n_cols.
        assert sorted(self.Zv.keys()) == sorted(self.outputs)
        assert len(self.Zv) == self.n_cols()
        assert len(self.dims()) == self.n_cols()
        # Nv should account for each column.
        assert sum(self.Nv()) == self.n_cols()
        # Nv should have an entry for each view.
        assert len(self.Nv()) == max(self.Zv.values())+1
        for v in xrange(len(self.Nv())):
            assert len(self.views[v].dims) == self.Nv(v)
            self.views[v]._check_partitions()
        # Dependence constraints.
        assert vu.validate_crp_constrained_partition(
            self.Zv, self.Cd, self.Ci, self.Rd, self.Ri)

    # --------------------------------------------------------------------------
    # Serialize

    def to_metadata(self):
        metadata = dict()

        # Dataset.
        # XXX FIXME
        metadata['X'] = np.asarray(self.X.values()).T.tolist()
        metadata['outputs'] = self.outputs

        # Iteration counts.
        metadata['iterations'] = self.iterations

        # View partition data.
        metadata['alpha'] = self.alpha
        metadata['Zv'] = [self.Zv[c] for c in self.outputs]

        # View data.
        metadata['Zrv'] = []
        metadata['view_alphas'] = []

        # Column data.
        metadata['cctypes'] = []
        metadata['hypers'] = []
        metadata['distargs'] = []

        for dim in self.dims():
            metadata['cctypes'].append(dim.cctype)
            metadata['hypers'].append(dim.hypers)
            metadata['distargs'].append(dim.distargs)

        for view in self.views:
            metadata['Zrv'].append([view.Zr[i] for i in sorted(view.Zr)])
            metadata['view_alphas'].append(view.alpha)

        return metadata

    def to_pickle(self, fileptr):
        metadata = self.to_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None: rng = gu.gen_rng(0)
        return cls(
            np.asarray(metadata['X']),
            outputs=metadata.get('outputs', None),
            cctypes=metadata.get('cctypes', None),
            distargs=metadata.get('distargs',None),
            Zv=metadata.get('Zv', None),
            Zrv=metadata.get('Zrv', None),
            alpha=metadata.get('alpha', None),
            view_alphas=metadata.get('view_alphas', None),
            hypers=metadata.get('hypers', None),
            iterations=metadata.get('iterations', None),
            rng=rng)

    @classmethod
    def from_pickle(cls, fileptr, rng=None):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata, rng=rng)


from gpmcc.crosscat import statedoc
statedoc.load_docstrings(sys.modules[__name__])
