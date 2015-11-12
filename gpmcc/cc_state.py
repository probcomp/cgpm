import copy
import numpy
import pylab
import random
from math import log
from math import exp
from scipy.stats import gamma
from numpy.random import exponential as exprnd

import gpmcc.utils.general as utils
import gpmcc.utils.cc_plot_utils as pu
import gpmcc.utils.sampling as su

from gpmcc.cc_types import normal_uc
from gpmcc.cc_types import beta_uc
from gpmcc.cc_types import normal
from gpmcc.cc_types import binomial
from gpmcc.cc_types import multinomial
from gpmcc.cc_types import lognormal
from gpmcc.cc_types import poisson
from gpmcc.cc_types import vonmises
from gpmcc.cc_types import vonmises_uc

from gpmcc.cc_view import cc_view
from gpmcc.cc_dim import cc_dim
from gpmcc.cc_dim_uc import cc_dim_uc

_is_uncollapsed = {
    'normal'      : False,
    'normal_uc'   : True,
    'beta_uc'     : True,
    'binomial'    : False,
    'multinomial' : False,
    'lognormal'   : False,
    'poisson'     : False,
    'vonmises'    : False,
    'vonmises_uc' : True,
    }

_cctype_class = {
    'normal'      : normal.Normal,
    'normal_uc'   : normal_uc.NormalUC,
    'beta_uc'     : beta_uc.BetaUC,
    'binomial'    : binomial.Binomial,
    'multinomial' : multinomial.Multinomial,
    'lognormal'   : lognormal.Lognormal,
    'poisson'     : poisson.Poisson,
    'vonmises'    : vonmises.Vonmises,
    'vonmises_uc' : vonmises_uc.VonmisesUC,
    }

_all_kernels = ['column_z','state_alpha','row_z','column_hypers','view_alphas']


class cc_state(object):
    """
    cc_state. The main crosscat object.
    properties:
    -- n_rows: (int) number of rows
    -- n_cols: (int) number of columns
    -- n_grid: (int) number of bins in hyperparameter grids
    """

    def __init__(self, X, cctypes, distargs, n_grid=30, Zv=None, Zrcv=None, hypers=None, seed=None):
        """
        cc_state constructor

        input arguments:
        -- X: a list of numpy data columns.
        -- cctypes: a list of strings where each entry is the data type for 
        each column.
        -- distargs: a list of distargs appropriate for each type in cctype.
        For details on distrags see the documentation for each data type.

        optional arguments:
        -- n_grid: number of bins for hyperparameter grids. Default = 30.
        -- Zv: The assignment of columns to views. If not specified, a 
        partition is generated randomly
        -- Zrcv: The assignment of rows to clusters for each view
        -- ct_kernel: which column transition kenerl to use. Default = 0 (Gibbs)
        -- seed: seed the random number generator. Default = system time.

        example:
        >>> import numpy
        >>> n_rows = 100
        >>> X = [numpy.random.normal(n_rows), numpy.random.normal(n_rows)]
        >>> State = cc_state(X, ['normal', 'normal'], [None, None])
        """

        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)

        self.n_rows = len(X[0])
        self.n_cols = len(X)
        self.n_grid = n_grid

        # construct the dims
        self.dims = []
        for col in range(self.n_cols):
            Y = X[col] 
            cctype = cctypes[col]
            if _is_uncollapsed[cctype]:
                dim = cc_dim_uc(Y, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs[col])
            else:
                dim = cc_dim(Y, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs[col])
            self.dims.append(dim)

        # set the hyperparameters in the dims
        if hypers is not None:
            for d in range(self.n_cols):
                self.dims[d].set_hypers(hypers[d])

        # initialize CRP alpha  
        self.alpha_grid = utils.log_linspace(1.0/self.n_cols, self.n_cols, self.n_grid)
        self.alpha = random.choice(self.alpha_grid)

        assert len(self.dims) == self.n_cols

        if Zrcv is not None:
            assert Zv is not None
            assert len(Zv) == self.n_cols
            assert len(Zrcv) == max(Zv)+1
            assert len(Zrcv[0]) == self.n_rows

        # construct the view partition
        if Zv is None:
            Zv, Nv, V = utils.crp_gen(self.n_cols, self.alpha)
        else:
            Nv = utils.bincount(Zv)
            V = len(Nv)

        # construct views
        self.views = []
        for view in range(V):
            indices = [i for i in range(self.n_cols) if Zv[i] == view]
            dims_view = []
            for index in indices:
                dims_view.append(self.dims[index])

            if Zrcv is None:
                self.views.append(cc_view(dims_view, n_grid=n_grid))
            else:
                self.views.append(cc_view(dims_view, Z=numpy.array(Zrcv[view]), n_grid=n_grid))

        self.Zv = numpy.array(Zv)
        self.Nv = Nv
        self.V = V

    @classmethod
    def from_metadata(cls, X, metadata):
        
        Zv = metadata['Zv']
        Zrcv = metadata['Zrcv']
        n_grid = metadata['n_grid']
        hypers = metadata['hypers']
        cctypes = metadata['cctypes']
        distargs = metadata['distargs']

        return cls(X, cctypes, distargs, n_grid, Zv, Zrcv, hypers)

    def get_metadata(self):
        metadata = dict()
        
        # misc data
        metadata['n_grid'] = self.n_grid

        # view data
        metadata['V'] = self.V
        metadata['Nv'] = self.Nv
        metadata['Zv'] = self.Zv

        # category data
        metadata['K'] = []
        metadata['Nk'] = []
        metadata['Zrcv'] = []

        # column data
        metadata['hypers'] = []
        metadata['cctypes'] = []
        metadata['distargs'] = []
        metadata['suffstats'] = []

        for dim in self.dims:
            metadata['hypers'].append(dim.hypers)
            metadata['distargs'].append(dim.distargs)
            metadata['cctypes'].append(dim.cctype)
            metadata['suffstats'].append(dim.get_suffstats())

        for view in self.views:
            metadata['K'].append(view.K)
            metadata['Nk'].append(view.Nk)
            metadata['Zrcv'].append(view.Z)

        return metadata
            

        
    def append_dim(self, X_f, cctype, distargs=None, ct_kernel=0, m=1):
        """
        Add a new data column to X.
        Inputs:
        -- X_f: a numpy array of data
        -- cctype: type of the data
        Keyword args:
        -- distargs: for multinomial data
        -- ct_kernel: must be 0 or 2. MH kernel cannot be used to append
        -- m: for ct_kernel=2. Number of auxiliary parameters
        """

        col = self.n_cols
        n_grid = self.n_grid

        if _is_uncollapsed[cctype]:
            dim = cc_dim_uc(X_f, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs)
        else:
            dim = cc_dim(X_f, _cctype_class[cctype], col, n_grid=n_grid, distargs=distargs)

        self.n_cols += 1

        self.dims.append(dim)
        self.Zv = numpy.append(self.Zv, -1)
        
        if _is_uncollapsed[cctype]:
            column_transition_kernel_collapsed(m=m, append=True)
        else:
            column_transition_kernel_uncollapsed(m=m, append=True)

        self.__check_partitions()

    def transition(self, N=1, kernel_list=None, ct_kernel=0, which_rows=None, which_cols=None, m=1, do_plot=False):
        """
        Do transitions.

        optional arguments:
        -- N: number of transitions.
        -- kernel_list: which kernels to do.
        -- ct_kernel: which column transitino kernel to use {0,1,2} = {Gibbs, MH, Aux Gibbs}
        -- which_rows: list of rows to apply the transitions to
        -- which_cols: list of columns to apply the transitions to
        -- do_plot: plot the state of the sampler (real-time)
        examples:
        >>> State.transition()
        >>> State.transition(N=100)
        >>> State.transition(N=100, kernel_list=['column_z','row_z'])
        """

        kernel_dict = {
            'column_z'      : lambda : self.__transition_columns(which_cols,ct_kernel,m=m),
            'state_alpha'   : lambda : self.__transition_state_alpha(),
            'row_z'         : lambda : self.__transition_rows(which_rows),
            'column_hypers' : lambda : self.__transition_column_hypers(which_rows),
            'view_alphas'   : lambda : self.__transition_view_alphas(),
        }

        if kernel_list is None:
            kernel_list = _all_kernels

        kernel_fns = [ kernel_dict[kernel] for kernel in kernel_list ]

        if do_plot:
            pylab.ion()
            layout = pu.get_state_plot_layout(self.n_cols)
            fig = pylab.figure(num=None, figsize=(layout['plot_inches_y'], layout['plot_inches_x']), 
                dpi=75, facecolor='w', edgecolor='k',frameon=False,tight_layout=True)
            self.__plot(fig,layout)

        for i in range(N):
            print i
            # random.shuffle(kernel_fns)
            for kernel in kernel_fns:
                kernel()
            if do_plot:
                self.__plot(fig,layout)

    def set_data(self, data):
        """
        Testing. Resets the suffstats in all clusters in all dims to reflect
        the new data.
        """
        for col in range(self.n_cols):
            self.dims[col].X = data[col]
            self.dims[col].reassign(self.views[self.Zv[col]].Z)

    def dump_data(self):
        """
        Testing. Clears the suffstats in all clusters in all dims.
        """
        for dim in self.dims:
            dim.dump_data()

    def _update_prior_grids(self):
        for dim in self.dims:
            dim.update_prior_grids()

    def __transition_columns(self, which_cols=None, ct_kernel=0, m=3):
        """Transition column assignment to views"""

        if which_cols is None:
            which_cols = [i for i in range(self.n_cols)]

        random.shuffle(which_cols)

        for col in which_cols:
            if self.dims[col].mode == 'collapsed':
                self.__transition_columns_kernel_collapsed(col, m=m, append=False)
            elif self.dims[col].mode == 'uncollapsed':
                self.__transition_columns_kernel_uncollapsed(col, m=m, append=False)
            else:
                raise ValueError('unsupported dim.mode for column %i: %s' % (col, self.dims[col].mode))



    def __transition_columns_kernel_collapsed(self, col, m=3, append=False):
        """Gibbs with auxiliary parameters for collapsed data types"""

        if append:
            col = self.n_cols-1

        # get start view, v_a, and check whether a singleton
        v_a = self.Zv[col]

        if append:
            is_singleton = False
            pv = list(self.Nv)
        else:
            is_singleton = (self.Nv[v_a] == 1)

            pv = list(self.Nv)
            # Get crp probabilities under each view. remove from current view.
            # If v_a is a singleton, do not consider move to new singleton view.
            if is_singleton:
                pv[v_a] = self.alpha
            else:
                pv[v_a] -= 1

        # take the log
        pv = numpy.log(numpy.array(pv))

        ps = []
        # calculate probability under each view's assignment
        dim = self.dims[col]

        for v in range(self.V):
            dim.reassign(self.views[v].Z)
            p_v = dim.full_marginal_logp()+pv[v]
            ps.append(p_v)

        # if not a singleton, propose m auxiliary parameters (views)
        if not is_singleton:
            # crp probability of singleton, split m times.
            log_aux = log(self.alpha/float(m))
            proposal_views = []
            for  _ in range(m):
                # propose (from prior) and calculate probability under each view
                proposal_view = cc_view([dim], n_grid=self.n_grid)
                proposal_views.append(proposal_view)
                dim.reassign(proposal_view.Z)
                p_v = dim.full_marginal_logp()+log_aux
                ps.append(p_v)

        # draw a view
        v_b = utils.log_pflip(ps)

        if append:
            if v_b >= self.V:
                index = v_b-self.V
                assert( index >= 0 and index < m)
                proposal_view = proposal_views[index]
            self.__append_new_dim_to_view(dim, v_b, proposal_view)
            return

        # clean up
        if v_b != v_a:
            if is_singleton:
                assert( v_b < self.V )
                self.__destroy_singleton_view(dim, v_a, v_b)
            elif v_b >= self.V:
                index = v_b-self.V
                assert( index >= 0 and index < m)
                proposal_view = proposal_views[index]
                self.__create_singleton_view(dim, v_a, proposal_view)
            else:
                self.__move_dim_to_view(dim, v_a, v_b)
        else:
            self.dims[col].reassign(self.views[v_a].Z)

        # self.__check_partitions()

    def __transition_columns_kernel_uncollapsed(self, col, m=3, append=False):
        """Gibbs with auxiliary parameters for uncollapsed data types"""

        if append:
            col = self.n_cols-1

        # get start view, v_a, and check whether a singleton
        v_a = self.Zv[col]

        if append:
            is_singleton = False
            pv = list(self.Nv)
        else:
            is_singleton = (self.Nv[v_a] == 1)

            pv = list(self.Nv)
            # Get crp probabilities under each view. remove from current view.
            # If v_a is a singleton, do not consider move to new singleton view.
            if is_singleton:
                pv[v_a] = self.alpha
            else:
                pv[v_a] -= 1

        # take the log
        pv = numpy.log(numpy.array(pv))

        ps = []
        # calculate probability under each view's assignment
        dim = self.dims[col]

        dim_holder = []

        for v in range(self.V):
            if v == v_a:
                dim_holder.append(dim)
            else:
                dim_holder.append(copy.deepcopy(dim))
                dim_holder[-1].reassign(self.views[v].Z)

            p_v = dim_holder[-1].full_marginal_logp()+pv[v]
            ps.append(p_v)

        # if not a singleton, propose m auxiliary parameters (views)
        if not is_singleton:
            # crp probability of singleton, split m times.
            log_aux = log(self.alpha/float(m))
            proposal_views = []
            for  _ in range(m):
                # propose (from prior) and calculate probability under each view
                dim_holder.append(copy.deepcopy(dim))

                proposal_view = cc_view([dim_holder[-1]], n_grid=self.n_grid)
                proposal_views.append(proposal_view)
                dim_holder[-1].reassign(proposal_view.Z)

                p_v = dim_holder[-1].full_marginal_logp()+log_aux
                ps.append(p_v)


        # draw a view
        v_b = utils.log_pflip(ps)

        newdim = dim_holder[v_b]
        self.dims[dim.index] = newdim

        if append:
            if v_b >= self.V:
                index = v_b-self.V
                assert( index >= 0 and index < m)
                proposal_view = proposal_views[index]
            self.__append_new_dim_to_view(newdim, v_b, proposal_view, is_uncollapsed=True)
            return

        # clean up
        if v_b != v_a:
            if is_singleton:
                assert( v_b < self.V )
                self.__destroy_singleton_view(newdim, v_a, v_b, is_uncollapsed=True)
            elif v_b >= self.V:
                index = v_b-self.V
                assert( index >= 0 and index < m)
                proposal_view = proposal_views[index]
                self.__create_singleton_view(newdim, v_a, proposal_view, is_uncollapsed=True)
            else:
                self.__move_dim_to_view(newdim, v_a, v_b, is_uncollapsed=True)

        # self.__check_partitions()

    def __transition_rows(self, which_rows=None):
        # move rows to new cluster
        for view in self.views:
            view.reassign_rows_to_cats(which_rows=which_rows)

    def __transition_column_hypers(self, which_cols=None):
        if which_cols is None:
            which_cols = range(self.n_cols)

        for i in which_cols:
            self.dims[i].update_hypers()

    def __transition_view_alphas(self):
        for view in self.views:
            view.transition_alpha()

    def __transition_state_alpha(self):

        logps = numpy.zeros(self.n_grid)
        for i in range(self.n_grid):
            alpha = self.alpha_grid[i]
            logps[i] = utils.unorm_lcrp_post(alpha, self.n_cols, self.V, lambda x: 0)
        # log_pdf_lambda = lambda a : utils.lcrp(self.n_cols, self.Nv, a) + self.alpha_prior_lambda(a)

        index = utils.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def __destroy_singleton_view(self, dim, to_destroy, move_to, is_uncollapsed=False):
        self.Zv[dim.index] = move_to
        self.views[to_destroy].release_dim(dim.index)
        zminus = numpy.nonzero(self.Zv>to_destroy)
        self.Zv[zminus] -= 1
        self.views[move_to].assimilate_dim(dim, is_uncollapsed=is_uncollapsed)
        self.Nv[move_to] += 1
        del self.Nv[to_destroy]
        del self.views[to_destroy]
        self.V -= 1

    def __create_singleton_view(self, dim, current_view_index, proposal_view, is_uncollapsed=False):
        self.Zv[dim.index] = self.V
        if not is_uncollapsed:
            dim.reassign(proposal_view.Z)
        self.views[current_view_index].release_dim(dim.index)
        self.Nv[current_view_index] -= 1
        self.Nv.append(1)
        self.views.append(proposal_view)
        self.V += 1

    def __move_dim_to_view(self, dim, move_from, move_to, is_uncollapsed=False):
        self.Zv[dim.index] = move_to
        self.views[move_from].release_dim(dim.index)
        self.Nv[move_from] -= 1
        self.views[move_to].assimilate_dim(dim, is_uncollapsed=is_uncollapsed)
        self.Nv[move_to] += 1

    def __append_new_dim_to_view(self, dim, append_to, proposal_view, is_uncollapsed=False):
        self.Zv[dim.index] = append_to
        if append_to == self.V:
            self.Nv.append(1)
            self.V += 1
            self.views.append(proposal_view)
        else:
            self.Nv[append_to] += 1
            self.views[append_to].assimilate_dim(dim, is_uncollapsed=is_uncollapsed)

        self.__check_partitions()

    def __plot(self, fig, layout):
        # do not plot more than 6 by 4
        if self.n_cols > 24:
            return

        pylab.clf()
        for dim in self.dims:
            index = dim.index
            ax = pylab.subplot(layout['plots_x'], layout['plots_y'], index)
            if self.Zv[index] >= len(layout['border_color']):
                border_color = 'gray'
            else:
                border_color = layout['border_color'][self.Zv[index]]
            dim.plot_dist()
            ax.spines['bottom'].set(lw=7,color=border_color)
            ax.spines['top'].set(lw=7,color=border_color)
            ax.spines['right'].set(lw=7,color=border_color)
            ax.spines['left'].set(lw=7,color=border_color)
            pylab.text(1,1, "K: %i " % len(dim.clusters),
                transform=ax.transAxes,
                fontsize=18, weight='bold', color='blue',
                horizontalalignment='right',verticalalignment='top')

        pylab.draw()

    # for debugging only
    def __check_partitions(self):
        # Nv should account for each column
        assert sum(self.Nv) == self.n_cols

        # Nv should have an entry for each view
        assert len(self.Nv) == self.V

        #
        assert max(self.Zv) == self.V-1

        for v in range(len(self.Nv)):
            # check that the number of dims actually assigned to the view matches 
            # the count in Nv
            assert len(self.views[v].dims) == self.Nv[v]

            Nk = self.views[v].Nk
            K = self.views[v].K

            assert sum(Nk) == self.n_rows
            assert len(Nk) == K
            assert max(self.views[v].Z) == K-1

            for dim in self.views[v].dims.values():
                # make sure the number of clusters in each dim in the view is the same
                # and is the same as described in the view (K, Nk)
                assert len(dim.clusters) == len(Nk)
                assert len(dim.clusters) == K
                for k in range(len(dim.clusters)):
                    assert dim.clusters[k].N == Nk[k]
