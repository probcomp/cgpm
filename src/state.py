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

import copy
import sys
import time
import cPickle as pickle
from math import log

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import logsumexp

import gpmcc.utils.general as gu
import gpmcc.utils.plots as pu
import gpmcc.utils.validation as vu

from gpmcc.view import View
from gpmcc.dim import Dim

_all_kernels = [
    'column_z',
    'state_alpha',
    'row_z',
    'column_hypers',
    'view_alphas'
    ]

class State(object):
    """State, the main crosscat object."""

    def __init__(self, X, cctypes, distargs=None, Zv=None, Zrv=None, alpha=None,
            view_alphas=None, hypers=None, Cd=None, Ci=None, Rd=None, Ri=None,
            rng=None):
        """Dim constructor provides a convenience method for bulk incorporate
        and unincorporate by specifying the data, and optinally view partition
        and row partition for each view.

        Parameters
        ----------
        X : np.ndarray
            A data matrix DxN, where D is the number of variabels and N is
            the number of observations.
        cctypes : list<str>
            Data type of each colum, see `utils.config` for valid cctypes.
        distargs : list<dict>, optional
            Distargs appropriate for each cctype in cctypes. For details on
            distargs see the documentation for each DistributionGpm. Empty
            distargs can be None or dict().
        Zv : list<int>, optional
            Assignmet of columns to views. If not specified a random
            partition is sampled.
        Zrv : list(list<int>), optional
            Assignment of rows to clusters in each view, where Zrv[k] is
            the Zr for View k. If not specified a random partition is
            sampled. If specified, then Zv must also be specified.
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
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        # Seed.
        self.rng = gu.gen_rng() if rng is None else rng

        # Dataset.
        self.X = np.asarray(X)

        # Distargs.
        if distargs is None:
            distargs = [None] * len(cctypes)

        # Constraints.
        self.Cd = [] if Cd is None else Cd
        self.Ci = [] if Ci is None else Ci
        self.Rd = {} if Rd is None else Rd
        self.Ri = {} if Ri is None else Ri
        if len(self.Cd) > 0:
            raise ValueError('Dependency constraints not yet implemented.')

        # Generate dimensions.
        dims = []
        for col in xrange(self.n_cols()):
            dim_hypers = None if hypers is None else hypers[col]
            D = Dim(cctypes[col], col, hypers=dim_hypers,
                distargs=distargs[col], rng=self.rng)
            D.transition_hyper_grids(self.X[:,col])
            dims.append(D)

        # Generate CRP alpha.
        self.alpha_grid = gu.log_linspace(1./self.n_cols(), self.n_cols(), 30)
        if alpha is None:
            alpha = self.rng.choice(self.alpha_grid)
        self.alpha = alpha

        # Generate view partition.
        if Zv is None:
            if len(self.Cd) + len(self.Ci) == 0:
                Zv = gu.simulate_crp(self.n_cols(), self.alpha, rng=self.rng)
            else:
                Zv = gu.simulate_crp_constrained(self.n_cols(), self.alpha,
                    self.Cd, self.Ci, self.Rd, self.Ri, rng=self.rng)
        self.Zv = list(Zv)

        # Generate views.
        self.views = []
        for v in sorted(set(self.Zv)):
            view_dims = [dims[i] for i in xrange(self.n_cols()) if Zv[i] == v]
            Zr = None if Zrv is None else np.asarray(Zrv[v])
            alpha = None if view_alphas is None else view_alphas[v]
            view = View(self.X, view_dims, Zr=Zr, alpha=alpha, rng=self.rng)
            self.views.append(view)

        self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, X, cctype, distargs=None, v=None):
        """Incorporate a new Dim into this State.

        Parameters
        ----------
        X : np.array
            An array of data with length self.n_rows().
        cctype : str
            DistributionGpm name see `gpmcc.utils.config`.
        distargs : dict, optional.
            Distargs appropriate for the cctype. For details on
            distargs see the documentation for each DistributionGpm.
        v : int, optional
            Index of the view to assign the data. If unspecified, will be
            sampled. If 0 <= v < len(state.views) then insert into an existing
            View. If v = len(state.views) then singleton view will be created
            with a partition from the CRP prior.
        """
        assert len(X) == self.n_rows()
        self.X = np.column_stack((self.X, X))

        col = self.n_cols() - 1
        D = Dim(cctype, col, distargs=distargs, rng=self.rng)
        D.transition_hyper_grids(self.X[:,col])

        for view in self.views:
            view.set_dataset(self.X)

        transition = [col] if v is None else []
        v = 0 if v is None else v

        if 0 <= v < self.n_views():
            view = self.views[v]
        elif v == self.n_views():
            view = View(self.X, [], rng=self.rng)
            self._append_view(view)

        view.incorporate_dim(D)
        self.Zv.append(v)

        self.transition_columns(cols=transition)
        self.transition_column_hypers(cols=[col])
        self._check_partitions()

    def unincorporate_dim(self, col):
        """Unincorporate the existing dim with index col."""
        if self.n_cols() == 1:
            raise ValueError('State has only one dim, cannot unincorporate.')

        D_all = self.dims()
        D_del = D_all[col]
        del D_all[col]

        v = self.Zv[col]
        self.views[v].unincorporate_dim(D_del)
        del self.Zv[col]

        if self.Nv(v) == 0:
            self._delete_view(v)

        for i, dim in enumerate(D_all):
            dim.index = i

        self.X = np.delete(self.X, col, 1)
        for view in self.views:
            view.set_dataset(self.X)
            view.reindex_dims()

        self._check_partitions()

    def incorporate_rows(self, X, k=None):
        """Incorporate list of new rows into global dataset X.

        Parameters
        ----------
        X : np.array
            A (r x self.n_cols) list of data, where r is the number of
            new rows to incorporate.
        k : list(list<int>), optional
            A (r x self.n_views()) list of integers, where r is the number of
            new rows to incorporate, and k[r][i] is the cluster to insert row r
            in view i. If k[r][i] is greater than the number of
            clusters in view[i] an error will be thrown. To specify cluster
            assignments for only some views, use None in all other locations
            i.e. k=[[None,2,None],[[0,None,1]]].
        """
        rowids = xrange(self.n_rows(), self.n_rows() + len(X))
        self.X = np.vstack((self.X, X))

        if k is None:
            k = [[None] * self.n_views()] * len(rowids)

        for v, view in enumerate(self.views):
            view.set_dataset(self.X)
            for r, rowid in enumerate(rowids):
                view.incorporate_row(rowid, k=k[r][v])

        self._check_partitions()

    def unincorporate_rows(self, rowids):
        """Unincorporate a list of rowids from dataset X. All r in rowids must
        be in range(0, State.n_rows())."""
        if self.n_rows() == 1:
            raise ValueError('State has only one row, cannot unincorporate.')

        self.X = np.delete(self.X, rowids, 0)

        for view in self.views:
            for rowid in rowids:
                view.unincorporate_row(rowid)
            view.set_dataset(self.X)
            view.reindex_rows()

        self._check_partitions()

    # --------------------------------------------------------------------------
    # Schema updates.

    def update_cctype(self, col, cctype, hypers=None, distargs=None):
        """Update the distribution type of self.dims[col] to cctype.

        Parameters
        ----------
        col : int
            Index of column to update.
        cctype : str
            DistributionGpm name see `gpmcc.utils.config`.
        distargs : dict, optional.
            Distargs appropriate for the cctype. For details on
            distargs see the documentation for each DistributionGpm.
        """
        # Obtain dimensions.
        D_old = self.dim_for(col)
        D_new = Dim(cctype, col, hypers=hypers, distargs=distargs, rng=self.rng)
        # Update views.
        self.view_for(col).unincorporate_dim(D_old)
        self.view_for(col).incorporate_dim(D_new)
        # Run transitions.
        self.transition_column_hyper_grids(cols=[col])
        self.transition_column_hypers(cols=[col])
        self.transition_column_params(cols=[col])
        # Confirm OK.
        self._check_partitions()

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence=None):
        """Compute density of query under the posterior predictive distirbution.

        Parameters
        ----------
        rowid : int
            The rowid of the member of the population to simulate from.
            If 0 <= rowid < state.n_rows then the latent variables of member
            rowid will be taken as conditioning variables.
            Otherwise logpdf for a hypothetical member is computed,
            marginalizing over latent variables.
        query : list(tuple<int>)
            A list of pairs (col, val) at which to query the logpdf.
        evidence : list(tuple<int>), optional
            A list of pairs (col, val) of observed values in the row to
            condition on

        Returns
        -------
        logpdf : float
            The logpdf(query|rowid, evidence).
        """
        if evidence is None:
            evidence = []

        vu.validate_query_evidence(
            self.X, rowid, self._is_hypothetical(rowid), query,
            evidence=evidence)

        logpdf = 0
        queries, evidences = self._get_view_qe(query, evidence)
        for v in queries:
            logpdf += self.views[v].logpdf(
                rowid, queries[v], evidences.get(v,[]))

        return logpdf

    def logpdf_bulk(self, rowids, queries, evidences=None):
        """Evaluate multiple queries at once, used by Engine."""
        if evidences is None:
            evidences = [[] for _ in xrange(len(rowids))]
        assert len(rowids) == len(queries) == len(evidences)
        logpdfs = []
        for rowid, query, evidence in zip(rowids, queries, evidences):
            logpdfs.append(self.logpdf(rowid, query, evidence))
        return np.asarray(logpdfs)

    def logpdf_marginal(self):
        return gu.logp_crp(len(self.Zv), self.Nv(), self.alpha) + \
            sum(v.logpdf_marginal() for v in self.views)

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, query, evidence=None, N=1):
        """Simulate from the posterior predictive distirbution.

        Parameters
        ----------
        rowid : int
            The rowid of the member of the population to simulate from.
            If 0 <= rowid < state.n_rows then the latent variables of member
            rowid will be taken as conditioning variables.
            Otherwise a hypothetical member is simulated, marginalizing over
            latent variables.
        query : list<int>
            A list of col numbers to simulate from.
        evidence : list(tuple<int>), optional
            A list of pairs (col, val) of observed values in the row to
            condition on.
        N : int, optional.
            Number of samples to return.

        Returns
        -------
        samples : np.array
            A N x len(query) array, where samples[i] ~ P(query|rowid, evidence).
        """
        if evidence is None:
            evidence = []

        vu.validate_query_evidence(
            self.X, rowid, self._is_hypothetical(rowid), query,
            evidence=evidence)

        samples = np.zeros((N, len(query)))
        queries, evidences = self._get_view_qe(query, evidence)
        for v in queries:
            v_query = queries[v]
            v_evidence = evidences.get(v, [])
            draws = self.views[v].simulate(rowid, v_query, v_evidence, N=N)
            for i, c in enumerate(v_query):
                samples[:,query.index(c)] = draws[:,i]

        return samples

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None):
        """Evaluate multiple queries at once, used by Engine."""
        if evidences is None:
            evidences = [[] for _ in xrange(len(rowids))]
        if Ns is None:
            Ns = [1 for _ in xrange(len(rowids))]
        assert len(rowids) == len(queries) == len(evidences) == len(Ns)
        samples = []
        for rowid, query, evidence, n in zip(rowids, queries, evidences, Ns):
            samples.append(self.simulate(rowid, query, evidence, n))
        return samples

    # --------------------------------------------------------------------------
    # Mutual information

    def mutual_information(self, col0, col1, evidence=None, N=1000):
        """Computes the mutual information MI(col0:col1|evidence).

        Mutual information with conditioning variables can be interpreted in two
        forms
            - MI(X:Y|Z=z): point-wise CMI, (this function).
            - MI(X:Y|Z): expected pointwise CMI E_Z[MI(X:Y|Z)] under Z.

        The rowid is hypothetical. For any observed member, the rowid is
        sufficient and decouples all columns.

        Parameters
        ----------
        col0, col1 : int
            Columns to comptue MI. If col0 = col1 then estimate of the entropy
            is returned.
        evidence : list(tuple<int>), optional
            A list of pairs (col, val) of observed values to condition on.
        N : int, optional.
            Number of samples to use in the Monte Carlo estimate.

        Returns
        -------
        mi : float
            A point estimate of the mutual information.
        """
        if evidence is None:
            evidence = []

        def samples_logpdf(cols, samples, evidence):
            queries = [zip(cols, samples[i]) for i in xrange(N)]
            return self.logpdf_bulk([-1]*N, queries, [evidence]*N)

        # MI or entropy?
        if col0 != col1:
            samples = self.simulate(-1, [col0, col1], evidence=evidence, N=N)
            PXY = samples_logpdf([col0, col1], samples, evidence)
            PX = samples_logpdf([col0], samples[:,0].reshape(-1,1), evidence)
            PY = samples_logpdf([col1], samples[:,1].reshape(-1,1), evidence)
            return (np.sum(PXY) - np.sum(PX) - np.sum(PY)) / N
        else:
            samples = self.simulate(-1, [col0], evidence=evidence, N=N)
            PX = samples_logpdf([col0], samples[:,0].reshape(-1,1), evidence)
            return - np.sum(PX) / N

    def conditional_mutual_information(self, col0, col1, evidence, T=100,
            N=1000):
        """Computes conditional mutual information MI(col0:col1|evidence).

        Mutual information with conditioning variables can be interpreted in two
        forms
            - MI(X:Y|Z=z): point-wise CMI.
            - MI(X:Y|Z): expected pointwise CMI E_z[MI(X:Y|Z=z)] under Z
            (this function).

        The rowid is hypothetical. For any observed member, the rowid is
        sufficient and decouples all columns.

        Parameters
        ----------
        col0, col1 : int
            Columns to comptue MI. If col0 = col1 then estimate of the entropy
            is returned.
        evidence : list<int>
            A list of columns to condition on.
        T : int, optional.
            Number of samples to use in external Monte Carlo estimate (z~Z).
        N : int, optional.
            Number of samples to use in internal Monte Carlo estimate.

        Returns
        -------
        mi : float
            A point estimate of the mutual information.
        """
        samples = self.simulate(-1, evidence, N=T)
        mi = 0
        for sample in samples:
            mi += self.mutual_information(
                col0, col1, evidence=zip(evidence, sample), N=N)
        return mi / T

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N=None, S=None, kernels=None, target_rows=None,
            target_cols=None, target_views=None, do_plot=False,
            do_progress=True):
        """Run all infernece kernels. For targeted inference, see other exposed
        inference commands.

        Parameters
        ----------
        N : int, optional
            Number of iterations to transition. Defaults to 1.
        S : float, optional
            Number of seconds to transition. If both N and S specified,
            the minimum is used.
        kernels : list<{'alpha', 'view_alphas', 'column_params', 'column_hypers'
                'rows', 'columns'}>, optional
            List of inference kernels to run in this inference transition.
            Default is all.
        target_views, target_rows, target_cols : list<int>, optional
            Views, rows and columns to apply the kernels. Default is all.
        do_plot : boolean, optional
            Plot the state of the sampler (real-time), 24 columns max. Only
            available when transition by iterations.
        do_progress : boolean, optional
            Show a progress bar for number of target iterations or elapsed time.
            If transition by time, may exceed 100%.

        Examples
        --------
        >>> State.transition()
        >>> State.transition(N=100)
        >>> State.transition(N=100, kernels=['rows', 'column_hypers'],
                target_cols=[1,2], target_rows=range(100))
        >>> State.transition(N=100, S=30)
        """
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
            plt.ion()
            plt.show()
            layout = pu.get_state_plot_layout(self.n_cols())
            fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
                layout['plot_inches_x']), dpi=75, facecolor='w',
                edgecolor='k', frameon=False, tight_layout=True)
            self._do_plot(fig, layout)

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
        logps = [gu.logp_crp_unorm(self.n_cols(), self.n_views(), alpha)
            for alpha in self.alpha_grid]
        index = gu.log_pflip(logps, rng=self.rng)
        self.alpha = self.alpha_grid[index]

    def transition_view_alphas(self, views=None):
        if views is None:
            views = xrange(self.n_views())
        for v in views:
            self.views[v].transition_alpha()

    def transition_column_params(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dim_for(c).transition_params()

    def transition_column_hypers(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dim_for(c).transition_hypers()

    def transition_column_hyper_grids(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dim_for(c).transition_hyper_grids(self.X[:,c])

    def transition_rows(self, views=None, rows=None):
        if self.n_rows() == 1:
            return
        if views is None:
            views = xrange(self.n_views())
        for v in views:
            self.views[v].transition_rows(rows=rows)

    def transition_columns(self, cols=None, m=2):
        """Transition column assignment to views."""
        if self.n_cols() == 1:
            return
        if cols is None:
            cols = range(self.n_cols())
        self.rng.shuffle(cols)
        for c in cols:
            self._transition_column(c, m)

    # --------------------------------------------------------------------------
    # Helpers

    def n_rows(self):
        return np.shape(self.X)[0]

    def n_cols(self):
        return np.shape(self.X)[1]

    def cctypes(self):
        return [d.cctype for d in self.dims()]

    def distargs(self):
        return [d.distargs for d in self.dims()]

    # --------------------------------------------------------------------------
    # Plotting

    def plot(self):
        """Plots observation histogram and posterior distirbution of dims."""
        layout = pu.get_state_plot_layout(self.n_cols())
        fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
            layout['plot_inches_x']), dpi=75, facecolor='w',
            edgecolor='k', frameon=False, tight_layout=True)
        self._do_plot(fig, layout)
        plt.show()

    # --------------------------------------------------------------------------
    # Internal

    def dim_for(self, c):
        """Dim object for column c."""
        return self.view_for(c).dims[c]

    def dims(self):
        """Dim d."""
        return [self.view_for(d).dims[d] for d in xrange(self.n_cols())]

    def view_for(self, d):
        """View object for Dim d."""
        return self.views[self.Zv[d]]

    def n_views(self):
        """Number of Views."""
        return len(self.views)

    def Nv(self, v=None):
        """Number of dims in View v."""
        if v is not None:
            return len(self.views[v].dims)
        return [len(view.dims) for view in self.views]

    def _transition_column(self, col, m):
        """Gibbs on col assignment to Views, with m auxiliary parameters"""
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
        vprops_aux = [View(self.X, [], rng=self.rng) for _ in m_aux]

        logp_data_aux = [get_data_logp(view, dim)
            for (view, dim) in zip(vprops_aux, dprops_aux)]

        # Extend data structs with auxiliary proposals.
        dprops.extend(dprops_aux)
        logp_data.extend(logp_data_aux)

        # Compute the CRP probabilities.
        logp_crp = gu.logp_crp_gibbs(self.Nv(), self.Zv, col, self.alpha, m)
        assert len(logp_data) == len(logp_crp)

        # Overall view probabilities.
        p_view = [d+c for (d,c) in zip(logp_data, logp_crp)]

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
        self.Zv = [i-1 if i>v else i for i in self.Zv]

    def _append_view(self, view):
        """Append a view and return and its index."""
        assert len(view.dims) == 0
        self.views.append(view)
        return self.n_views()-1

    def _is_hypothetical(self, rowid):
        return not 0 <= rowid < self.n_rows()

    def _get_view_qe(self, query, evidence):
        """queries[v], evidences[v] are the queries, evidences for view v."""
        queries, evidences = {}, {}
        for q in query:
            col = q if isinstance(q, int) else q[0]
            if self.Zv[col] in queries:
                queries[self.Zv[col]].append(q)
            else:
                queries[self.Zv[col]] = [q]
        for e in evidence:
            col = e[0]
            if self.Zv[col] in evidences:
                evidences[self.Zv[col]].append(e)
            else:
                evidences[self.Zv[col]] = [e]
        return queries, evidences

    def _do_plot(self, fig, layout):
        # Do not plot more than 6 by 4.
        if self.n_cols() > 24:
            return
        fig.clear()
        for dim in self.dims():
            index = dim.index
            ax = fig.add_subplot(layout['plots_x'], layout['plots_y'], index+1)
            if self.Zv[index] >= len(layout['border_color']):
                border_color = 'gray'
            else:
                border_color = layout['border_color'][self.Zv[index]]
            dim.plot_dist(self.X[:,dim.index], ax=ax)
            ax.text(1,1, "K: %i " % len(dim.clusters),
                transform=ax.transAxes, fontsize=12, weight='bold',
                color='blue', horizontalalignment='right',
                verticalalignment='top')
        plt.draw()

    def _do_progress(self, percentage):
        progress = ' ' * 30
        fill = int(percentage * len(progress))
        progress = '[' + '=' * fill + progress[fill:] + ']'
        print '{} {:1.2f}%\r'.format(progress, 100 * percentage),
        sys.stdout.flush()

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha > 0.
        # Zv and dims should match n_cols.
        assert len(self.Zv) == self.n_cols()
        assert len(self.dims()) == self.n_cols()
        # Nv should account for each column.
        assert sum(self.Nv()) == self.n_cols()
        # Nv should have an entry for each view.
        assert len(self.Nv()) == max(self.Zv)+1
        for v in xrange(len(self.Nv())):
            # Check that the number of dims actually assigned to the view
            # matches the count in Nv.
            assert len(self.views[v].dims) == self.Nv(v)
            Nk = self.views[v].Nk
            assert len(self.views[v].Zr) == sum(Nk) == self.n_rows()
            assert max(self.views[v].Zr) == len(Nk)-1
            for dim in self.views[v].dims.values():
                # Ensure number of clusters in each dim in views[v]
                # is the same and as described in the view (K, Nk).
                assert len(dim.clusters) == len(Nk)
                for k in xrange(len(dim.clusters)):
                    rowids = [r for (r, z) in enumerate(self.views[v].Zr)
                        if z == k]
                    num_nans = np.sum(np.isnan(self.X[rowids,dim.index]))
                    assert dim.clusters[k].N == Nk[k] - num_nans
        # Dependence constraints.
        assert vu.validate_crp_constrained_partition(
            self.Zv, self.Cd, self.Ci, self.Rd, self.Ri)

    # --------------------------------------------------------------------------
    # Serialize

    def to_metadata(self):
        metadata = dict()

        # Dataset.
        metadata['X'] = self.X.tolist()

        # Entropy.
        metadata['rng'] = self.rng

        # View partition data.
        metadata['alpha'] = self.alpha
        metadata['Zv'] = self.Zv

        # View data.
        metadata['Nk'] = []
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
            metadata['Nk'].append(view.Nk)
            metadata['Zrv'].append(view.Zr)
            metadata['view_alphas'].append(view.alpha)

        return metadata

    def to_pickle(self, fileptr):
        metadata = self.to_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_metadata(cls, metadata):
        X = np.asarray(metadata['X'])
        if 'seed' in metadata:  # XXX Backward compatability.
            metadata['rng'] = gu.gen_rng(metadata['seed'])
        return cls(X, metadata['cctypes'], metadata['distargs'],
            Zv=metadata['Zv'], Zrv=metadata['Zrv'], alpha=metadata['alpha'],
            view_alphas=metadata['view_alphas'], hypers=metadata['hypers'],
            rng=metadata['rng'])

    @classmethod
    def from_pickle(cls, fileptr):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)
