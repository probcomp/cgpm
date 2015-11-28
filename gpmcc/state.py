# -*- coding: utf-8 -*-

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import copy
import sys
from math import log

import numpy as np
import matplotlib.pyplot as plt

import gpmcc.utils.general as gu
import gpmcc.utils.plots as pu
import gpmcc.utils.config as cu

from gpmcc.view import View
from gpmcc.dim import Dim

_all_kernels = ['column_z','state_alpha','row_z','column_hypers','view_alphas']

class State(object):
    """State. The main crosscat object.

    Attributes:
    -- n_rows: (int) number of rows.
    -- n_cols: (int) number of columns.
    -- n_grid: (int) number of bins in hyperparameter grids.
    """

    def __init__(self, X, cctypes, distargs, n_grid=30, Zv=None, Zrcv=None,
            hypers=None, seed=None):
        """State constructor.

        Input arguments:
        -- X: A tranposed data matrix DxN, where D is the number of variables
            and N is the number of observations.
        -- cctypes: a list of strings where each entry is the data type for
            each column.
        -- distargs: a list of distargs appropriate for each type in cctype.
            For details on distrags see the documentation for each data type.

        Optional arguments:
        -- n_grid: number of bins for hyperparameter grids. Default = 30.
        -- Zv: The assignment of columns to views. If not specified, a
            partition is generated randomly
        -- Zrcv: The assignment of rows to clusters for each view
        -- ct_kernel: which column transition kenerl to use. Default = 0 (Gibbs)
        -- seed: seed the random number generator. Default = system time.

        Example:
        >>> import np
        >>> n_rows = 100
        >>> X = [np.random.normal(n_rows), np.random.normal(n_rows)]
        >>> state = State(X, ['normal', 'normal'], [None, None])
        """
        self.seed = 0 if seed is None else seed
        np.random.seed(self.seed)

        self.n_rows = len(X[0])
        self.n_cols = len(X)
        self.n_grid = n_grid

        self.dims = []
        for col in xrange(self.n_cols):
            Y = X[col]
            cctype = cctypes[col]
            mode = 'uncollapsed' if cu.is_uncollapsed(cctype) else 'collapsed'
            dim_hypers = None if hypers is None else hypers[col]
            dim = Dim(Y, cctype, col, n_grid=n_grid, hypers=dim_hypers,
                mode=mode, distargs=distargs[col])
            self.dims.append(dim)

        # Initialize CRP alpha.
        self.alpha_grid = gu.log_linspace(1.0 / self.n_cols, self.n_cols,
            self.n_grid)
        self.alpha = np.random.choice(self.alpha_grid)

        assert len(self.dims) == self.n_cols

        if Zrcv is not None:
            assert Zv is not None
            assert len(Zv) == self.n_cols
            assert len(Zrcv) == max(Zv)+1
            assert len(Zrcv[0]) == self.n_rows

        # Construct the view partition.
        if Zv is None:
            Zv, Nv, V = gu.crp_gen(self.n_cols, self.alpha)
        else:
            Nv = gu.bincount(Zv)
            V = len(Nv)

        # Construct views.
        self.views = []
        for view in range(V):
            indices = [i for i in range(self.n_cols) if Zv[i] == view]
            dims_view = []
            for index in indices:
                dims_view.append(self.dims[index])
            Zr = None if Zrcv is None else np.asarray(Zrcv[view])
            view = View(dims_view, Zr=Zr, n_grid=n_grid)
            self.views.append(view)

        self.X = X
        self.Zv = np.array(Zv)
        self.Nv = Nv

    # def append_dim(self, X_f, cctype, distargs=None, ct_kernel=0, m=1):
    #     """Add a new data column to X.

    #     Input arguments:
    #     -- X_f: a np array of data
    #     -- cctype: type of the data

    #     Optional arguments:
    #     -- distargs: for multinomial data
    #     -- ct_kernel: must be 0 or 2. MH kernel cannot be used to append
    #     -- m: for ct_kernel=2. Number of auxiliary parameters
    #     """
    #     col = self.n_cols
    #     n_grid = self.n_grid
    #     mode = 'collapsed'
    #     if cu.is_uncollapsed(cctype):
    #         mode = 'uncollapsed'
    #     dim = Dim(X_f, cu.dist_class(cctype), col, n_grid=n_grid,
    #         mode=mode, distargs=distargs)
    #     self.n_cols += 1
    #     self.dims.append(dim)
    #     self.Zv = np.append(self.Zv, -1)
    #     if cu.is_uncollapsed(cctype):
    #         self._transition_columns_kernel_collapsed(-1, m=m, append=True)
    #     else:
    #         self._transition_columns_kernel_uncollapsed(-1, m=m, append=True)
    #     self._check_partitions()

    def transition(self, N=1, kernel_list=None, ct_kernel=0, target_rows=None,
            target_cols=None, m=1, do_plot=False):
        """Do transitions.

        Optional arguments:
        -- N: number of transitions.
        -- kernel_list: which kernels to do.
        -- ct_kernel: which column transition kernel to use {0,1,2}
        --      = {Gibbs, MH, Aux Gibbs}
        -- target_rows: list of rows to apply the transitions to
        -- target_cols: list of columns to apply the transitions to
        -- do_plot: plot the state of the sampler (real-time)

        Examples:
        >>> State.transition()
        >>> State.transition(N=100)
        >>> State.transition(N=100, kernel_list=['column_z','row_z'])
        """
        kernel_dict = {
            'column_z' :
                lambda : self._transition_columns(target_cols, ct_kernel, m=m),
            'state_alpha':
                lambda : self._transition_state_alpha(),
            'row_z':
                lambda : self._transition_rows(target_rows),
            'column_hypers' :
                lambda : self._transition_column_hypers(target_rows),
            'view_alphas'   :
                lambda : self._transition_view_alphas(),
        }

        if kernel_list is None:
            kernel_list = _all_kernels

        kernel_fns = [kernel_dict[kernel] for kernel in kernel_list]

        if do_plot:
            plt.ion()
            plt.show()
            layout = pu.get_state_plot_layout(self.n_cols)
            fig = plt.figure(num=None, figsize=(layout['plot_inches_y'],
                layout['plot_inches_x']), dpi=75, facecolor='w', edgecolor='k',
                frameon=False,tight_layout=True)
            self._plot(fig, layout)

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
                self._plot(fig, layout)
                plt.pause(0.0001)
        print

    def set_data(self, data):
        """Testing. Resets the suffstats in all clusters in all dims to reflect
        the new data.
        """
        for col in range(self.n_cols):
            self.dims[col].X = data[col]
            self.dims[col].reassign(self.views[self.Zv[col]].Zr)

    def clear_data(self):
        """Clears the suffstats in all clusters in all dims."""
        for view in self.views:
            view.clear_data()

    def _update_prior_grids(self):
        for dim in self.dims:
            dim.update_prior_grids()

    def _transition_columns(self, target_cols=None, ct_kernel=0, m=3):
        """Transition column assignment to views."""
        if target_cols is None:
            target_cols = [i for i in range(self.n_cols)]

        np.random.shuffle(target_cols)

        for col in target_cols:
            if self.dims[col].mode == 'collapsed':
                self._transition_columns_kernel_collapsed(col, m=m)
            elif self.dims[col].mode == 'uncollapsed':
                self._transition_columns_kernel_uncollapsed(col, m=m)
            else:
                raise ValueError('unsupported dim.mode for column %i: %s' \
                    % (col, self.dims[col].mode))

    def _transition_columns_kernel_collapsed(self, col, m=3):
        """Gibbs with auxiliary parameters for collapsed data types."""
        v_a = self.Zv[col]
        is_singleton = (self.Nv[v_a] == 1)
        p_crp = self._compute_view_crp_logps(v_a)

        # Calculate probability under each view's assignment
        p_view = []
        dim = self.dims[col]
        for v in xrange(len(self.Nv)):
            dim.reassign(self.views[v].Zr)
            # Total view prob is data + crp.
            p_view_v = dim.full_marginal_logp() + p_crp[v]
            p_view.append(p_view_v)

        # If not a singleton, propose m auxiliary parameters (views)
        if not is_singleton:
            # Crp probability of singleton, split m times.
            p_crp_aux = log(self.alpha/float(m))
            proposal_views = []
            for  _ in range(m):
                # Propose (from prior) and calculate probability under each view
                proposal_view = View([dim], n_grid=self.n_grid)
                proposal_views.append(proposal_view)
                dim.reassign(proposal_view.Zr)
                p_view_aux = dim.full_marginal_logp() + p_crp_aux
                p_view.append(p_view_aux)

        # Draw a view.
        v_b = gu.log_pflip(p_view)

        # Clean up.
        if len(self.Nv) <= v_b:
            index = v_b - len(self.Nv)
            assert 0 <= index and index < m
            proposal_view = proposal_views[index]
            self._create_singleton_view(dim, v_a, proposal_view)
        else:
            if is_singleton:
                assert v_b < len(self.Nv)
            self._move_dim_to_view(dim, v_a, v_b)

        self._check_partitions()

    def _transition_columns_kernel_uncollapsed(self, col, m=3):
        """Gibbs with auxiliary parameters for uncollapsed data types"""
        v_a = self.Zv[col]
        is_singleton = (self.Nv[v_a] == 1)
        p_crp = self._compute_view_crp_logps(v_a)

        ps = []
        # Calculate column probability under each view's assignment.
        dim = self.dims[col]
        dim_holder = []
        for v in xrange(len(self.Nv)):
            if v == v_a:
                dim_holder.append(dim)
            else:
                dim_holder.append(copy.deepcopy(dim))
                dim_holder[-1].reassign(self.views[v].Zr)
            p_v = dim_holder[-1].full_marginal_logp() + p_crp[v]
            ps.append(p_v)

        # If not a singleton, propose m auxiliary parameters (views)
        if not is_singleton:
            # Crp probability of singleton, split m times.
            log_aux = log(self.alpha/float(m))
            proposal_views = []
            for  _ in range(m):
                # propose (from prior) and calculate probability under each view
                dim_holder.append(copy.deepcopy(dim))
                proposal_view = View([dim_holder[-1]], n_grid=self.n_grid)
                proposal_views.append(proposal_view)
                dim_holder[-1].reassign(proposal_view.Zr)
                p_v = dim_holder[-1].full_marginal_logp() + log_aux
                ps.append(p_v)

        # Draw a view
        v_b = gu.log_pflip(ps)
        new_dim = dim_holder[v_b]
        self.dims[dim.index] = new_dim

        # Are we moving to a singleton?
        if len(self.Nv) <= v_b:
            index = v_b - len(self.Nv)
            assert 0 <= index and index < m
            proposal_view = proposal_views[index]
            self._create_singleton_view(new_dim, v_a, proposal_view)
        # Move new_dim to the new view.
        else:
            if is_singleton:
                assert v_b < len(self.Nv)
            self._move_dim_to_view(new_dim, v_a, v_b)

        # self._check_partitions()

    def _transition_rows(self, target_rows=None):
        # move rows to new cluster
        for view in self.views:
            view.transition_rows(target_rows=target_rows)

    def _transition_column_hypers(self, target_cols=None):
        if target_cols is None:
            target_cols = range(self.n_cols)

        for i in target_cols:
            self.dims[i].transition_hypers()

    def _transition_view_alphas(self):
        for view in self.views:
            view.transition_alpha()

    def _transition_state_alpha(self):
        logps = np.zeros(self.n_grid)
        for i in range(self.n_grid):
            alpha = self.alpha_grid[i]
            logps[i] = gu.unorm_lcrp_post(alpha, self.n_cols, len(self.Nv),
                lambda x: 0)
        # log_pdf_lambda = lambda a : gu.lcrp(self.n_cols, self.Nv, a) +
        # self.alpha_prior_lambda(a)
        index = gu.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def _create_singleton_view(self, dim, current_view_index, proposal_view,
            is_uncollapsed=False):
        self.Zv[dim.index] = len(self.Nv)
        if not is_uncollapsed:
            dim.reassign(proposal_view.Zr)
        self.views[current_view_index].release_dim(dim.index)
        self.Nv[current_view_index] -= 1
        self.Nv.append(1)
        self.views.append(proposal_view)

    def _move_dim_to_view(self, dim, move_from, move_to, is_uncollapsed=False):
        self.Zv[dim.index] = move_to
        self.views[move_from].release_dim(dim.index)
        self.Nv[move_from] -= 1
        self.views[move_to].assimilate_dim(dim)
        self.Nv[move_to] += 1
        # If move_from was a singleton, destroy.
        if self.Nv[move_from] == 0:
            # Decrement view index of all other views.
            zminus = np.nonzero(self.Zv>move_from)
            self.Zv[zminus] -= 1
            del self.Nv[move_from]
            del self.views[move_from]

    def _append_new_dim_to_view(self, dim, append_to, proposal_view):
        self.Zv[dim.index] = append_to
        if append_to == len(self.Nv):
            self.Nv.append(1)
            self.views.append(proposal_view)
        else:
            self.Nv[append_to] += 1
            self.views[append_to].assimilate_dim(dim)
        self._check_partitions()

    def _compute_view_crp_logps(self, view):
        is_singleton = (self.Nv[view] == 1)
        p_crp = list(self.Nv)
        if is_singleton:
            p_crp[view] = self.alpha
        else:
            p_crp[view] -= 1
        return np.log(np.asarray(p_crp))

    def _plot(self, fig, layout):
        # do not plot more than 6 by 4
        if self.n_cols > 24:
            return
        fig.clear()
        for dim in self.dims:
            index = dim.index
            ax = fig.add_subplot(layout['plots_x'], layout['plots_y'], index)
            if self.Zv[index] >= len(layout['border_color']):
                border_color = 'gray'
            else:
                border_color = layout['border_color'][self.Zv[index]]
            dim.plot_dist(ax=ax)
            ax.text(1,1, "K: %i " % len(dim.clusters),
                transform=ax.transAxes,
                fontsize=12, weight='bold', color='blue',
                horizontalalignment='right',verticalalignment='top')
        plt.draw()

    def _check_partitions(self):
        # for debugging only
        # Nv should account for each column
        assert sum(self.Nv) == self.n_cols
        # Nv should have an entry for each view
        assert max(self.Zv) == len(self.Nv)-1
        for v in range(len(self.Nv)):
            # check that the number of dims actually assigned to the view
            # matches the count in Nv
            assert len(self.views[v].dims) == self.Nv[v]
            Nk = self.views[v].Nk
            K = self.views[v].K
            assert sum(Nk) == self.n_rows
            assert len(Nk) == K
            assert max(self.views[v].Zr) == K-1
            for dim in self.views[v].dims.values():
                # make sure the number of clusters in each dim in the view is the same
                # and is the same as described in the view (K, Nk)
                assert len(dim.clusters) == len(Nk)
                assert len(dim.clusters) == K
                for k in range(len(dim.clusters)):
                    assert dim.clusters[k].N == Nk[k]

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
        metadata['K'] = []
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
            metadata['K'].append(view.K)
            metadata['Nk'].append(view.Nk)
            metadata['Zrcv'].append(view.Zr)

        return metadata

    def to_pickle(self, fileptr):
        import pickle
        metadata = self.get_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_pickle(cls, fileptr):
        import pickle
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)

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
