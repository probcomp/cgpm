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
import pickle
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
    """State. The main crosscat object."""

    def __init__(self, X, cctypes, distargs, n_grid=30, Zv=None, Zrcv=None,
            hypers=None, seed=None):
        """State constructor.

        Parameters
        ----------
        X : np.ndarray
            A data matrix DxN, where D is the number of variabels and N is
            the number of observations.
        cctypes : list<str>
            Data type of each colum, see `utils.config` for valid cctypes.
        distargs : list
            Distargs appropriate for each cctype in cctypes. For details on
            distargs see the documentation for each DistributionGpm.
        n_grid : int, optional
            Number of bins for hyperparameter grids.
        Zv : list<int>, optional
            Assignmet of columns to views. If not specified a random
            partition is sampled.
        Zrcv : list, optional
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
        self.n_rows, self.n_cols = np.shape(X)
        self.cctypes = cctypes
        self.distargs = distargs

        # Hyperparameters.
        self.n_grid = n_grid

        # Construct dimensions.
        self.dims = []
        for col in xrange(self.n_cols):
            dim_hypers = None if hypers is None else hypers[col]
            self.dims.append(
                Dim(X[:,col], cctypes[col], col, n_grid=n_grid,
                hypers=dim_hypers, distargs=distargs[col]))

        # Initialize CRP alpha.
        self.alpha_grid = gu.log_linspace(1./self.n_cols, self.n_cols,
            self.n_grid)
        self.alpha = np.random.choice(self.alpha_grid)

        # Construct view partition.
        if Zv is None:
            Zv, Nv, V = gu.simulate_crp(self.n_cols, self.alpha)
        else:
            Nv = list(np.bincount(Zv))
            V = len(Nv)
        self.Zv = np.array(Zv)
        self.Nv = Nv

        # Construct views.
        self.views = []
        for v in xrange(V):
            dims = [self.dims[i] for i in xrange(self.n_cols) if Zv[i] == v]
            Zr = None if Zrcv is None else np.asarray(Zrcv[v])
            self.views.append(View(self.X, dims, Zr=Zr, n_grid=n_grid))

        self._check_partitions()

    # --------------------------------------------------------------------------
    # Observe

    def incorporate_dim(self, X, cctype, distargs=None, v=None):
        """Incorporate a new Dim into the StateGPM.

        Parameters
        ----------
        X : np.array
            An array of data with length `n_rows`.
        cctype : list<str>
            Data type of the column, see `utils.config` for valid cctypes.
        distargs : dict, optional.
            Distargs appropriate for the cctype. For details on
            distargs see the documentation for each DistributionGpm.
        v : int, optional
            Index of the view to assign the data. If Zv unspecified, will be
            sampled from the CRP. If 0 <= Zv < len(state.Nv) then will insert
            into an existing if. If Zv = len(state.Nv) a singleton view will be
            created sampled from the prior.
        """
        assert len(X) == self.n_rows
        self.X = np.column_stack((self.X, X))
        self.n_rows, self.n_cols = np.shape(self.X)

        col = self.n_cols - 1
        self.dims.append(Dim(X, cctype, col, n_grid=self.n_grid,
            distargs=distargs))

        for view in self.views:
            view.set_dataset(self.X)

        if 0 <= v < len(self.Nv):
            self.views[v].incorporate_dim(self.dims[-1])
            self.Zv = np.append(self.Zv, v)
            self.Nv[v] += 1
        elif v == len(self.Nv):
            self.views.append(
                View(self.X, [self.dims[-1]], n_grid=self.n_grid))
            self.Zv = np.append(self.Zv, v)
            self.Nv.append(1)
        else:
            self.views[0].incorporate_dim(self.dims[-1])
            self.Zv = np.append(self.Zv, 0)
            self.Nv[0] += 1
            self.transition_columns(target_cols=[col])

        self.transition_column_hypers(target_cols=[col])
        self._check_partitions()

    def unincorporate_dim(self, col):
        """Unincorporate an existing dim.

        Parameters
        ----------
        col : int
            Index of the dim to unincorporate.
        """
        self.X = np.delete(self.X, col, 1)
        self.n_rows, self.n_cols = np.shape(self.X)

        v = self.Zv[col]
        self.views[v].unincorporate_dim(self.dims[col])
        self.Nv[v] -= 1
        self.Zv = np.delete(self.Zv, col)

        if self.Nv[v] == 0:
            zminus = np.nonzero(self.Zv>v)
            self.Zv[zminus] -= 1
            del self.Nv[v]
            del self.views[v]

        del self.dims[col]
        for i, dim in enumerate(self.dims):
            dim.index = i

        for view in self.views:
            view.set_dataset(self.X)
            view.reindex_dims()

        self._check_partitions()

    def incorporate_row(self, X):
        raise ValueError('Cannot incorporate row yet.')

    def unincorporate_row(self, X):
        raise ValueError('Cannot unincorporate row yet.')

    # --------------------------------------------------------------------------
    # logpdf

    def logpdf(self, rowid, query, evidence=None):
        # XXX Implement logpdf unobserved.
        return self.logpdf_unobserved(query, evidence=evidence)

    def logpdf_unobserved(self, query, evidence=None, N=1):
        """Simulates a hypothetical member, with no observed latents."""
        # Default parameter.
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

    # --------------------------------------------------------------------------
    # Simulate

    def simulate(self, rowid, query, evidence=None, N=1):
        # XXX Implement simulate unobserved.
        return self.simulate_unobserved(query, evidence=evidence, N=N)

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
                x = self.dims[col].simulate(sampled_k[v])
                draw.append(x)
            samples.append(draw)

        return np.asarray(samples)

    # --------------------------------------------------------------------------
    # Inference

    def transition(self, N=1, target_rows=None, target_cols=None,
            target_views=None, do_plot=False, do_progress=True):
        """Run all infernece kernels. For targeted inference, see other exposed
        inference commands.

        Parameters
        ----------
        N : int, optional
            Number of transitions.
        target_views, target_rows, target_cols : list<int>, optional
            Views, rows and columns to apply the kernels. Default is all.
        do_plot : boolean, optional
            Plot the state of the sampler (real-time).
        do_progress : boolean, optional
            Show a progress bar for number of target iterations (real-time).

        Examples
        --------
        >>> State.transition()
        >>> State.transition(N=100)
        >>> State.transition(N=100, cols=[1,2], rows=range(100))
        """
        if do_progress:
                percentage = 0
                progress = ' ' * 30
                fill = int(percentage * len(progress))
                progress = '[' + '=' * fill + progress[fill:] + ']'
                print '{} {:1.2f}%\r'.format(progress, 100 * percentage),
                sys.stdout.flush()
        if do_plot:
            plt.ion()
            plt.show()
            layout = pu.get_state_plot_layout(self.n_cols)
            fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
                layout['plot_inches_x']), dpi=75, facecolor='w',
                edgecolor='k', frameon=False, tight_layout=True)
            self._do_plot(fig, layout)

        for i in xrange(N):
            self.transition_alpha()
            self.transition_view_alphas(target_views=target_views)
            self.transition_column_hypers(target_cols=target_cols)
            self.transition_rows(target_views=target_views,
                target_rows=target_rows)
            self.transition_columns(target_cols=target_cols)
            if do_progress:
                percentage = float(i+1) / N
                progress = ' ' * 30
                fill = int(percentage * len(progress))
                progress = '[' + '=' * fill + progress[fill:] + ']'
                print '{} {:1.2f}%\r'.format(progress, 100 * percentage),
                sys.stdout.flush()
            if do_plot:
                self._do_plot(fig, layout)
                plt.pause(1e-4)
        print

    def transition_alpha(self):
        logps = [gu.logp_crp_unorm(self.n_cols, len(self.Nv), alpha) for alpha
            in self.alpha_grid]
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def transition_view_alphas(self, target_views=None):
        if target_views is None:
            target_views = self.views
        for view in target_views:
            view.transition_alpha()

    def transition_column_hypers(self, target_cols=None):
        if target_cols is None:
            target_cols = range(self.n_cols)
        for i in target_cols:
            self.dims[i].transition_hypers()

    def transition_rows(self, target_views=None, target_rows=None):
        if target_views is None:
            target_views = self.views
        for view in target_views:
            view.transition_rows(target_rows=target_rows)

    def transition_columns(self, target_cols=None, m=2):
        """Transition column assignment to views."""
        if target_cols is None:
            target_cols = range(self.n_cols)
        np.random.shuffle(target_cols)
        for col in target_cols:
            self._transition_column(col, m)

    # --------------------------------------------------------------------------
    # Plotting

    def plot(self):
        """Plots sample histogram and learned distribution for each dim."""
        layout = pu.get_state_plot_layout(self.n_cols)
        fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
            layout['plot_inches_x']), dpi=75, facecolor='w',
            edgecolor='k', frameon=False, tight_layout=True)
        self._do_plot(fig, layout)
        plt.show()

    # --------------------------------------------------------------------------
    # Internal

    def _transition_column(self, col, m):
        """Gibbs with auxiliary parameters. Currently resampled uncollapsed
        parameters as a side-effect. m should be at least 1."""
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

        def get_propsal_dim(dim, view):
            if dim.is_collapsed() or view == v_a:
                return dim
            return copy.deepcopy(dim)

        # Calculate probability under existing view assignments.
        p_view = []
        proposal_dims = []
        for v in xrange(len(self.views)):
            proposal_dims.append(get_propsal_dim(self.dims[col], v))
            if v != v_a or self.dims[col].is_collapsed():
                proposal_dims[-1].reassign(self.X[:,col], self.views[v].Zr)
            p_view.append(proposal_dims[-1].marginal_logp() + p_crp[v])

        # Propose auxiliary views.
        p_crp_aux = log(self.alpha/float(m))
        proposal_views = []
        for _ in xrange(m-1 if singleton else m):
            proposal_dims.append(get_propsal_dim(self.dims[col], None))
            proposal_views.append(
                View(self.X, [proposal_dims[-1]], n_grid=self.n_grid))
            p_view.append(proposal_dims[-1].marginal_logp() + p_crp_aux)

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
        self.views[v_b].incorporate_dim(self.dims[col],
            reassign=self.dims[col].is_collapsed())
        self.Zv[col] = v_b
        self.Nv[v_a] -= 1
        self.Nv[v_b] += 1

        # Delete empty view?
        if self.Nv[v_a] == 0:
            zminus = np.nonzero(self.Zv>v_a)
            self.Zv[zminus] -= 1
            del self.Nv[v_a]
            del self.views[v_a]

        self._check_partitions()

    def _compute_cluster_crp_logps(self, view):
        """Returns a list of log probabilities that a new row joins each of the
        clusters in self.views[view], including a singleton."""
        log_crp_numer = np.log(self.views[view].Nk + [self.views[view].alpha])
        logp_crp_denom = log(self.n_rows + self.views[view].alpha)
        return log_crp_numer - logp_crp_denom

    def _compute_cluster_data_logps(self, col, x):
        """Returns a list of log probabilities that a new row for self.dims[col]
        obtains value x for each of the clusters in self.Zr[col], including a
        singleton."""
        return [self.dims[col].predictive_logp(x,k) for k in
            xrange(len(self.dims[col].clusters)+1)]

    def _do_plot(self, fig, layout):
        # Do not plot more than 6 by 4.
        if self.n_cols > 24:
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

    def _check_partitions(self):
        # For debugging only.
        assert self.alpha > 0.
        # Zv and dims should match n_cols.
        assert len(self.Zv) == self.n_cols
        assert len(self.dims) == self.n_cols
        # Nv should account for each column.
        assert sum(self.Nv) == self.n_cols
        # Nv should have an entry for each view.
        assert len(self.Nv) == max(self.Zv)+1
        for v in xrange(len(self.Nv)):
            # Check that the number of dims actually assigned to the view
            # matches the count in Nv.
            assert len(self.views[v].dims) == self.Nv[v]
            Nk = self.views[v].Nk
            assert sum(Nk) == self.n_rows
            assert max(self.views[v].Zr) == len(Nk)-1
            for dim in self.views[v].dims.values():
                # Ensure number of clusters in each dim in views[v]
                # is the same and as described in the view (K, Nk).
                assert len(dim.clusters) == len(Nk)
                for k in xrange(len(dim.clusters)):
                    rowids = np.where(self.views[v].Zr==k)[0]
                    num_nans = np.sum(np.isnan(self.X[rowids,dim.index]))
                    assert dim.clusters[k].N == Nk[k] - num_nans

    # --------------------------------------------------------------------------
    # Serialize

    def get_metadata(self):
        metadata = dict()

        # Dataset.
        metadata['X'] = self.X

        # Misc data.
        metadata['n_grid'] = self.n_grid
        metadata['seed'] = self.seed

        # View data.
        metadata['Nv'] = self.Nv
        metadata['Zv'] = self.Zv

        # Category data.
        metadata['Nk'] = []
        metadata['Zrcv'] = []

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

        return metadata

    def to_pickle(self, fileptr):
        metadata = self.get_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_metadata(cls, metadata):
        X = metadata['X']
        Zv = metadata['Zv']
        Zrcv = metadata['Zrcv']
        n_grid = metadata['n_grid']
        hypers = metadata['hypers']
        cctypes = metadata['cctypes']
        distargs = metadata['distargs']
        return cls(X, cctypes, distargs, n_grid=n_grid, Zv=Zv, Zrcv=Zrcv,
            hypers=hypers)

    @classmethod
    def from_pickle(cls, fileptr):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)
