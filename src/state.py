# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

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

    def __init__(self, X, cctypes, distargs=None, Zv=None, Zrcv=None, alpha=None,
            view_alphas=None, hypers=None, n_grid=30, seed=None):
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
            distargs can be None or {}.
        n_grid : int, optional
            Number of bins for hyperparameter grids.
        Zv : list<int>, optional
            Assignmet of columns to views. If not specified a random
            partition is sampled.
        Zrcv : list(list<int>), optional
            Assignment of rows to clusters in each view, where Zrcv[k] is
            the Zr for View k. If not specified a random partition is
            sampled. If specified, then Zv must also be specified.
        seed : int
            Seed the random number generator.
        """
        # Seed.
        self.seed = 0 if seed is None else seed
        np.random.seed(self.seed)

        # Dataset.
        self.X = np.asarray(X)

        # Hyperparameters.
        self.n_grid = n_grid

        # Distargs.
        if distargs is None:
            distargs = [None] * len(cctypes)

        # Generate dimensions.
        self.dims = []
        for col in xrange(self.n_cols()):
            dim_hypers = None if hypers is None else hypers[col]
            self.dims.append(
                Dim(X[:,col], cctypes[col], col, n_grid=n_grid,
                hypers=dim_hypers, distargs=distargs[col]))

        # Generate CRP alpha.
        self.alpha_grid = gu.log_linspace(1./self.n_cols(), self.n_cols(),
            self.n_grid)
        if alpha is None:
            alpha = np.random.choice(self.alpha_grid)
        self.alpha = alpha

        # Generate view partition.
        if Zv is None:
            Zv = gu.simulate_crp(self.n_cols(), self.alpha)
        self.Zv = list(Zv)
        self.Nv = list(np.bincount(Zv))

        # Generate views.
        self.views = []
        for v in xrange(len(self.Nv)):
            dims = [self.dims[i] for i in xrange(self.n_cols()) if Zv[i] == v]
            Zr = None if Zrcv is None else np.asarray(Zrcv[v])
            alpha = None if view_alphas is None else view_alphas[v]
            V = View(self.X, dims, Zr=Zr, alpha=alpha, n_grid=n_grid)
            self.views.append(V)

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
            sampled. If 0 <= v < len(state.Nv) then will insert into an existing
            View. If v = len(state.Nv) a singleton view will be created with a
            partition from the prior.
        """
        assert len(X) == self.n_rows()
        self.X = np.column_stack((self.X, X))

        col = self.n_cols() - 1
        self.dims.append(Dim(X, cctype, col, n_grid=self.n_grid,
            distargs=distargs))

        for view in self.views:
            view.set_dataset(self.X)

        if 0 <= v < len(self.Nv):
            self.views[v].incorporate_dim(self.dims[-1])
            self.Zv.append(v)
            self.Nv[v] += 1
        elif v == len(self.Nv):
            self.views.append(
                View(self.X, [self.dims[-1]], n_grid=self.n_grid))
            self.Zv.append(v)
            self.Nv.append(1)
        else:
            self.views[0].incorporate_dim(self.dims[-1])
            self.Zv.append(0)
            self.Nv[0] += 1
            self.transition_columns(cols=[col])

        self.transition_column_hypers(cols=[col])
        self._check_partitions()

    def unincorporate_dim(self, col):
        """Unincorporate the existing dim with index col."""
        if self.n_cols() == 1:
            raise ValueError('State has only one dim, cannot unincorporate.')

        self.X = np.delete(self.X, col, 1)

        v = self.Zv[col]
        self.views[v].unincorporate_dim(self.dims[col])
        self.Nv[v] -= 1
        del self.Zv[col]

        if self.Nv[v] == 0:
            self.Zv = [i-1 if i>v else i for i in self.Zv]
            del self.Nv[v]
            del self.views[v]

        del self.dims[col]
        for i, dim in enumerate(self.dims):
            dim.index = i

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
            A (r x len(self.views)) list of integers, where r is the number of
            new rows to incorporate, and k[r][i] is the cluster to insert row r
            in view i. If k[r][i] is greater than the number of
            clusters in view[i] an error will be thrown. To specify cluster
            assignments for only some views, use None in all other locations
            i.e. k=[[None,2,None],[[0,None,1]]].
        """
        rowids = xrange(self.n_rows(), self.n_rows() + len(X))
        self.X = np.vstack((self.X, X))

        if k is None:
            k = [[None] * len(self.views)] * len(rowids)

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
        if not 0 <= rowid < self.n_rows():
            return self.logpdf_unobserved(query, evidence=evidence)

        logpdf = 0
        for (col, val) in query:
            k = self.views[self.Zv[col]].Zr[rowid]
            logpdf += self.dims[col].logpdf(val, k)
        return logsumexp(logpdf)

    def logpdf_unobserved(self, query, evidence=None):
        """Simulates a hypothetical member, with no observed latents."""
        if evidence is None:
            evidence = []

        # Obtain all views of the query columns.
        query_views = set([self.Zv[col] for (col, _) in query])

        # Obtain the probability of hypothetical row belonging to each cluster.
        cluster_logps_for = dict()
        for v in query_views:
            # CRP densities.
            logp_crp = self._compute_cluster_crp_logps(v)
            # Evidence densities.
            logp_data = np.zeros(len(logp_crp))
            for (col, val) in evidence:
                if self.Zv[col] == v:
                    logp_data += self._compute_cluster_data_logps(col, val)
            cluster_logps_for[v] = gu.log_normalize(logp_crp+logp_data)

        logpdf = 0
        for (col, val) in query:
            # Query densities.
            logpdf += logsumexp(self._compute_cluster_data_logps(col, val)
                + cluster_logps_for[self.Zv[col]])

        return logpdf

    def logpdf_bulk(self, rowids, queries, evidences=None):
        """Evaluate multiple queries at once, used by Engine."""
        assert len(rowids) == len(queries) == len(evidences)
        if evidences is None:
            evidences = [[] for _ in xrange(len(rowids))]
        logpdfs = []
        for rowid, query, evidence in zip(rowids, queries, evidences):
            logpdfs.append(self.logpdf(rowid, query, evidence))
        return logpdfs

    def logpdf_marginal(self):
        return gu.logp_crp(len(self.Zv), self.Nv, self.alpha) + \
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
        if not 0 <= rowid < self.n_rows():
            return self.simulate_unobserved(query, evidence=evidence, N=N)

        samples = []
        for _ in xrange(N):
            draw = []
            for col in query:
                k = self.views[self.Zv[col]].Zr[rowid]
                x = self.dims[col].simulate(k)
                draw.append(x)
            samples.append(draw)
        return np.asarray(samples)

    def simulate_unobserved(self, query, evidence=None, N=1):
        """Simulates a hypothetical member, with no observed latents."""
        # Default parameter.
        if evidence is None:
            evidence = []

        # Obtain views of query columns.
        query_views = set([self.Zv[col] for col in query])

        # Obtain probability of hypothetical row belonging to each cluster.
        cluster_logps_for = dict()
        for v in query_views:
            # CRP densities.
            logp_crp = self._compute_cluster_crp_logps(v)
            # Evidence densities.
            logp_data = np.zeros(len(logp_crp))
            for (col, val) in evidence:
                if self.Zv[col] == v:
                    logp_data += self._compute_cluster_data_logps(col, val)
            cluster_logps_for[v] = gu.log_normalize(logp_crp+logp_data)

        samples = []
        for _ in xrange(N):
            sampled_k = dict()
            draw = []
            for v in query_views:
                # Sample cluster.
                sampled_k[v] = gu.log_pflip(cluster_logps_for[v])
            for col in query:
                # Sample data.
                k = sampled_k[self.Zv[col]]
                x = self.dims[col].simulate(k)
                draw.append(x)
            samples.append(draw)

        return np.asarray(samples)

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None):
        """Evaluate multiple queries at once, used by Engine."""
        assert len(rowids) == len(queries) == len(evidences) == len(Ns)
        if evidences is None:
            evidences = [[] for _ in xrange(len(rowids))]
        if Ns is None:
            Ns = [1 for _ in xrange(len(rowids))]
        samples = []
        for rowid, query, evidence, n in zip(rowids, queries, evidences, Ns):
            samples.append(self.simulate(rowid, query, evidence, n))
        return samples

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N=1, S=None, kernels=None, target_rows=None,
            target_cols=None, target_views=None, do_plot=False,
            do_progress=True):
        """Run all infernece kernels. For targeted inference, see other exposed
        inference commands.

        Parameters
        ----------
        N : int, optional
            Number of transitions.
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
        """
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

        # Transition by time.
        if S:
            start = time.time()
            if do_progress:
                self._do_progress(0)
            while True:
                for k in kernels:
                    _kernel_lookup[k]()
                    elapsed = time.time() - start
                    if elapsed >= S:
                        if do_progress:
                            self._do_progress(elapsed/S)
                        print
                        return
                if do_progress:
                    self._do_progress(elapsed/S)

        # Transition by iterations.
        if do_progress:
            self._do_progress(0)
        if do_plot:
            plt.ion()
            plt.show()
            layout = pu.get_state_plot_layout(self.n_cols())
            fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
                layout['plot_inches_x']), dpi=75, facecolor='w',
                edgecolor='k', frameon=False, tight_layout=True)
            self._do_plot(fig, layout)
        for i in xrange(N):
            for k in kernels:
                _kernel_lookup[k]()
            if do_progress:
                self._do_progress(float(i+1)/N)
            if do_plot:
                self._do_plot(fig, layout)
                plt.pause(1e-4)
        print

    def transition_alpha(self):
        logps = [gu.logp_crp_unorm(self.n_cols(), len(self.Nv), alpha) for alpha
            in self.alpha_grid]
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def transition_view_alphas(self, views=None):
        if views is None:
            views = xrange(len(self.views))
        for v in views:
            self.views[v].transition_alpha()

    def transition_column_params(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dims[c].transition_params()

    def transition_column_hypers(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dims[c].transition_hypers()

    def transition_column_hyper_grids(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dims[c].transition_hyper_grids(self.X[:,c], self.n_grid)

    def transition_rows(self, views=None, rows=None):
        if self.n_rows() == 1:
            return
        if views is None:
            views = xrange(len(self.views))
        for v in views:
            self.views[v].transition_rows(rows=rows)

    def transition_columns(self, cols=None, m=2):
        """Transition column assignment to views."""
        if self.n_cols() == 1:
            return
        if cols is None:
            cols = range(self.n_cols())
        np.random.shuffle(cols)
        for c in cols:
            self._transition_column(c, m)

    # --------------------------------------------------------------------------
    # Helpers

    def n_rows(self):
        return np.shape(self.X)[0]

    def n_cols(self):
        return np.shape(self.X)[1]

    def cctypes(self):
        return [self.dims[i].cctype for i in xrange(self.n_cols())]

    def distargs(self):
        return [self.dims[i].distargs for i in xrange(self.n_cols())]

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

    def _transition_column(self, col, m):
        """Gibbs on col assignment to Views, with m auxiliary parameters"""
        v_a = self.Zv[col]
        singleton = (self.Nv[v_a] == 1)

        # Compute CRP probabilities.
        p_crp = list(self.Nv)
        v_a = self.Zv[col]
        if self.Nv[v_a] == 1:
            p_crp[v_a] = self.alpha / float(m)
        else:
            p_crp[v_a] -= 1
        p_crp = np.log(p_crp)

        # Reuse collapsed, deepcopy uncollapsed.
        def get_propsal_dim(dim, view):
            if dim.is_collapsed() or view == v_a:
                return dim
            return copy.deepcopy(dim)

        # Calculate probability under existing views.
        p_view = []
        proposal_dims = []
        for v in xrange(len(self.views)):
            D = get_propsal_dim(self.dims[col], v)
            if v == v_a:
                logp = self.views[v].unincorporate_dim(D)
                self.views[v].incorporate_dim(D, reassign=D.is_collapsed())
            else:
                logp = self.views[v].incorporate_dim(D)
                self.views[v].unincorporate_dim(D)
            p_view.append(logp + p_crp[v])
            proposal_dims.append(D)

        # Propose auxiliary views.
        p_crp_aux = log(self.alpha/float(m))
        proposal_views = []
        for _ in xrange(m-1 if singleton else m):
            D = get_propsal_dim(self.dims[col], None)
            V = View(self.X, [], n_grid=self.n_grid)
            logp = V.incorporate_dim(D)
            p_view.append(logp + p_crp_aux)
            proposal_dims.append(D)
            proposal_views.append(V)

        # Draw view.
        v_b = gu.log_pflip(p_view)
        self.dims[col] = proposal_dims[v_b]

        # Append auxiliary view?
        if v_b >= len(self.views):
            self.views.append(proposal_views[v_b-len(self.Nv)])
            self.Nv.append(0)
            v_b = len(self.Nv)-1

        # Accounting.
        if v_a != v_b:
            self.views[v_a].unincorporate_dim(self.dims[col])
        self.views[v_b].incorporate_dim(
            self.dims[col], reassign=self.dims[col].is_collapsed())
        self.Zv[col] = v_b
        self.Nv[v_a] -= 1
        self.Nv[v_b] += 1

        # Delete empty view?
        if self.Nv[v_a] == 0:
            self.Zv = [i-1 if i>v_a else i for i in self.Zv]
            del self.Nv[v_a]
            del self.views[v_a]

        self._check_partitions()

    def _compute_cluster_crp_logps(self, view):
        """Returns a list of log probabilities that a new row joins each of the
        clusters in self.views[view], including a singleton."""
        log_crp_numer = np.log(self.views[view].Nk + [self.views[view].alpha])
        logp_crp_denom = log(self.n_rows() + self.views[view].alpha)
        return log_crp_numer - logp_crp_denom

    def _compute_cluster_data_logps(self, col, x):
        """Returns a list of log probabilities that a new row for self.dims[col]
        obtains value x for each of the clusters in self.Zr[col], including a
        singleton."""
        return [self.dims[col].logpdf(x,k) for k in
            xrange(len(self.dims[col].clusters)+1)]

    def _do_plot(self, fig, layout):
        # Do not plot more than 6 by 4.
        if self.n_cols() > 24:
            return
        fig.clear()
        for dim in self.dims:
            index = dim.index
            ax = fig.add_subplot(layout['plots_x'], layout['plots_y'],
                index+1)
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
        assert len(self.dims) == self.n_cols()
        # Nv should account for each column.
        assert sum(self.Nv) == self.n_cols()
        # Nv should have an entry for each view.
        assert len(self.Nv) == max(self.Zv)+1
        for v in xrange(len(self.Nv)):
            # Check that the number of dims actually assigned to the view
            # matches the count in Nv.
            assert len(self.views[v].dims) == self.Nv[v]
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

    # --------------------------------------------------------------------------
    # Serialize

    def to_metadata(self):
        metadata = dict()

        # Dataset.
        metadata['X'] = self.X

        # Misc data.
        metadata['n_grid'] = self.n_grid
        metadata['seed'] = self.seed

        # View partition data.
        metadata['alpha'] = self.alpha
        metadata['Nv'] = self.Nv
        metadata['Zv'] = self.Zv

        # View data.
        metadata['Nk'] = []
        metadata['Zrcv'] = []
        metadata['view_alphas'] = []

        # Column data.
        metadata['hypers'] = []
        metadata['cctypes'] = []
        metadata['distargs'] = []
        metadata['suffstats'] = []

        for dim in self.dims:
            metadata['hypers'].append(dim.hypers)
            metadata['cctypes'].append(dim.cctype)
            metadata['distargs'].append(dim.distargs)
            metadata['suffstats'].append(dim.get_suffstats())

        for view in self.views:
            metadata['Nk'].append(view.Nk)
            metadata['Zrcv'].append(view.Zr)
            metadata['view_alphas'].append(view.alpha)

        return metadata

    def to_pickle(self, fileptr):
        metadata = self.get_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_metadata(cls, metadata):
        return cls(metadata['X'], metadata['cctypes'], metadata['distargs'],
            Zv=metadata['Zv'], Zrcv=metadata['Zrcv'], alpha=metadata['alpha'],
            view_alphas=metadata['view_alphas'], hypers=metadata['hypers'],
            n_grid=metadata['n_grid'], seed=metadata['seed'])

    @classmethod
    def from_pickle(cls, fileptr):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)
