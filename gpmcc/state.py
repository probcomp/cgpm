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

import sys
import pickle
from math import log

import numpy as np
import matplotlib.pyplot as plt

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
            Zv, Nv, V = gu.crp_gen(self.n_cols, self.alpha)
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

    def transition(self, N=1, kernel_list=None, target_rows=None,
            target_cols=None, target_views=None, m=1, do_plot=False):
        """Run inference transitions.

        Parameters
        ----------
        N : int, optional
            Number of transitions.
        kernel_list : {_all_kernels}, optional.
            List of kernels to transition through. Default is all.
        target_views, target_rows, target_cols : list<int>, optional
            Views, rows and columns to apply the kernels. Default is all.
        do_plot : boolean, optional
            Plot the state of the sampler (real-time).

        Examples
        --------
        >>> State.transition()
        >>> State.transition(N=100)
        >>> State.transition(N=100, kernel_list=['column_z','row_z'],
        ...     cols=[1,2], rows=range(100))
        """
        kernel_dict = {
            'column_z' :
                lambda : self._transition_columns(target_cols=target_cols, m=m),
            'state_alpha' :
                lambda : self._transition_state_alpha(),
            'row_z' :
                lambda : self._transition_rows(target_views=target_views,
                    target_rows=target_rows),
            'column_hypers' :
                lambda : self._transition_column_hypers(target_cols=target_cols),
            'view_alphas' :
                lambda : self._transition_view_alphas(target_views=target_views),
        }

        if kernel_list is None:
            kernel_list = _all_kernels

        kernel_fns = [kernel_dict[kernel] for kernel in kernel_list]

        if do_plot:
            plt.ion()
            plt.show()
            layout = pu.get_state_plot_layout(self.n_cols)
            fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
                layout['plot_inches_x']), dpi=75, facecolor='w',
                edgecolor='k', frameon=False, tight_layout=True)
            self._do_plot(fig, layout)

        for i in xrange(N):
            percentage = float(i+1) / N
            progress = ' ' * 30
            fill = int(percentage * len(progress))
            progress = '[' + '=' * fill + progress[fill:] + ']'
            print '{} {:1.2f}%\r'.format(progress, 100 * percentage),
            sys.stdout.flush()
            for kernel in kernel_fns:
                kernel()
            if do_plot:
                self._do_plot(fig, layout)
                plt.pause(1e-4)
        print

    def plot(self):
        """Plots sample histogram and learned distribution for each dim."""
        layout = pu.get_state_plot_layout(self.n_cols)
        fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
            layout['plot_inches_x']), dpi=75, facecolor='w',
            edgecolor='k', frameon=False, tight_layout=True)
        self._do_plot(fig, layout)
        plt.show()

    def _transition_rows(self, target_views=None, target_rows=None):
        if target_views is None:
            target_views = self.views
        for view in target_views:
            view.transition_rows(target_rows=target_rows)

    def _transition_view_alphas(self, target_views=None):
        if target_views is None:
            target_views = self.views
        for view in target_views:
            view.transition_alpha()

    def _transition_state_alpha(self):
        logps = np.zeros(self.n_grid)
        for i in range(self.n_grid):
            alpha = self.alpha_grid[i]
            logps[i] = gu.unorm_lcrp_post(alpha, self.n_cols, len(self.Nv),
                lambda x: 0)
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def _transition_column_hypers(self, target_cols=None):
        if target_cols is None:
            target_cols = range(self.n_cols)
        for i in target_cols:
            self.dims[i].transition_hypers()

    def _transition_columns(self, target_cols=None, m=3):
        """Transition column assignment to views."""
        if target_cols is None:
            target_cols = range(self.n_cols)
        np.random.shuffle(target_cols)
        for col in target_cols:
            self._transition_column(col, m=m)

    def _transition_column(self, col, m=3):
        """Gibbs with auxiliary parameters. Currently resampled uncollapsed
        parameters as a side-effect."""
        dim = self.dims[col]
        v_a = self.Zv[col]
        singleton = (self.Nv[v_a] == 1)
        p_crp = self._compute_view_crp_logps(v_a)

        # XXX Major hack. Save logp under current view assignment.
        va_marginal_logp = self.dims[col].marginal_logp()

        # Calculate probability under each view's assignment
        p_view = []
        proposal_dims = []
        for v in xrange(len(self.Nv)):
            proposal_dims.append(dim)
            proposal_dims[-1].reassign(self.X[:,dim.index],
                self.views[v].Zr)
            p_view_v = dim.marginal_logp() + p_crp[v]
            # XXX Major hack continued,
            if v == v_a:
                p_view_v = va_marginal_logp + p_crp[v]
            p_view.append(p_view_v)

        # If not a singleton, propose m auxiliary parameters (views)
        if not singleton:
            # CRP probability of singleton, split m times.
            p_crp_aux = log(self.alpha/float(m))
            proposal_views = []
            for  _ in range(m):
                proposal_dims.append(dim)
                proposal_view = View(self.X, [proposal_dims[-1]],
                    n_grid=self.n_grid)
                proposal_views.append(proposal_view)
                p_view_aux = dim.marginal_logp() + p_crp_aux
                p_view.append(p_view_aux)

        # Draw a view.
        v_b = gu.log_pflip(p_view)
        self.dims[dim.index] = proposal_dims[v_b]

        # Register the dim with the new view.
        if len(self.Nv) <= v_b:
            index = v_b - len(self.Nv)
            assert 0 <= index and index < m
            self._create_singleton_view(dim, v_a, proposal_views[index])
        else:
            self._move_dim_to_view(dim, v_a, v_b)

        self._check_partitions()

    def _create_singleton_view(self, dim, current_view_index, proposal_view):
        self.Zv[dim.index] = len(self.Nv)
        dim.reassign(self.X[:,dim.index], proposal_view.Zr)
        self.views[current_view_index].remove_dim(dim.index)
        self.Nv[current_view_index] -= 1
        self.Nv.append(1)
        self.views.append(proposal_view)

    def _move_dim_to_view(self, dim, move_from, move_to):
        self.Zv[dim.index] = move_to
        self.views[move_from].remove_dim(dim.index)
        self.Nv[move_from] -= 1
        self.views[move_to].insert_dim(dim)
        self.Nv[move_to] += 1
        # If move_from was a singleton, destroy.
        if self.Nv[move_from] == 0:
            # Decrement view index of all other views.
            zminus = np.nonzero(self.Zv>move_from)
            self.Zv[zminus] -= 1
            del self.Nv[move_from]
            del self.views[move_from]

    def _compute_view_crp_logps(self, view):
        p_crp = list(self.Nv)
        if self.Nv[view] == 1:
            p_crp[view] = self.alpha
        else:
            p_crp[view] -= 1
        return np.log(np.asarray(p_crp))

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
                # for k in xrange(len(dim.clusters)):
                    # assert dim.clusters[k].N == Nk[k]

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
