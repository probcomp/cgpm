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
            view_alphas=None, hypers=None, Cd=None, Ci=None, seed=None):
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
            List of marginal dependence constraints. Each element in the list
            is a list of columns which are to be in the same view. Each column
            can only be in one such list i.e. [[1,2,5],[1,5]] is not allowed.
        Ci : list(tuple<int>), optional
            List of marginal independence constraints. Each element in the list
            is a 2-tuple of columns that must be independent, i.e.
            [(1,2),(1,3)].
        seed : int
            Seed the random number generator.
        """
        # Seed.
        self.seed = 0 if seed is None else seed
        np.random.seed(self.seed)

        # Dataset.
        self.X = np.asarray(X)

        # Distargs.
        if distargs is None:
            distargs = [None] * len(cctypes)

        # Constraints.
        self.Cd = [] if Cd is None else Cd
        self.Ci = [] if Ci is None else Ci
        if len(self.Cd) > 0:
            raise ValueError('Dependency constraints not yet implemented.')

        # Generate dimensions.
        dims = []
        for col in xrange(self.n_cols()):
            dim_hypers = None if hypers is None else hypers[col]
            D = Dim(cctypes[col], col, hypers=dim_hypers, distargs=distargs[col])
            D.transition_hyper_grids(self.X[:,col])
            dims.append(D)

        # Generate CRP alpha.
        self.alpha_grid = gu.log_linspace(1./self.n_cols(), self.n_cols(), 30)
        if alpha is None:
            alpha = np.random.choice(self.alpha_grid)
        self.alpha = alpha

        # Generate view partition.
        if Zv is None:
            if len(self.Cd) + len(self.Ci) == 0:
                Zv = gu.simulate_crp(self.n_cols(), self.alpha)
            else:
                Zv = gu.simulate_crp_constrained(
                    self.n_cols(), self.alpha, self.Cd, self.Ci)
        self.Zv = list(Zv)
        self.Nv = list(np.bincount(Zv))

        # Generate views.
        self.views = []
        for v in xrange(len(self.Nv)):
            view_dims = [dims[i] for i in xrange(self.n_cols()) if Zv[i] == v]
            Zr = None if Zrv is None else np.asarray(Zrv[v])
            alpha = None if view_alphas is None else view_alphas[v]
            V = View(self.X, view_dims, Zr=Zr, alpha=alpha)
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
        D = Dim(cctype, col, distargs=distargs)
        D.transition_hyper_grids(self.X[:,col])

        for view in self.views:
            view.set_dataset(self.X)

        if 0 <= v < len(self.Nv):
            self.views[v].incorporate_dim(D)
            self.Zv.append(v)
            self.Nv[v] += 1
        elif v == len(self.Nv):
            self.views.append(View(self.X, [D]))
            self.Zv.append(v)
            self.Nv.append(1)
        else:
            self.views[0].incorporate_dim(D)
            self.Zv.append(0)
            self.Nv[0] += 1
            self.transition_columns(cols=[col])

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
        self.Nv[v] -= 1
        del self.Zv[col]

        if self.Nv[v] == 0:
            self.Zv = [i-1 if i>v else i for i in self.Zv]
            del self.Nv[v]
            del self.views[v]

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
        # Contradictory base measures.
        if self.dims(col0).is_numeric() != self.dims(col1).is_numeric():
            raise ValueError('Cannot compute MI of numeric and symbolic.')
        if self.dims(col0).is_continuous() != self.dims(col1).is_continuous():
            raise ValueError('Cannot compute MI of continuous and discrete.')

        if evidence is None:
            evidence = []

        def samples_logpdf(cols, samples, evidence):
            queries = [zip(cols, samples[i]) for i in xrange(len(samples))]
            return self.logpdf_bulk(
                [-1]*len(samples), queries, [evidence]*(len(samples)))

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
            self.dims(c).transition_params()

    def transition_column_hypers(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dims(c).transition_hypers()

    def transition_column_hyper_grids(self, cols=None):
        if cols is None:
            cols = xrange(self.n_cols())
        for c in cols:
            self.dims(c).transition_hyper_grids(self.X[:,c])

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

    def dims(self, col=None):
        if col is not None:
            return self.views[self.Zv[col]].dims[col]
        return [self.views[self.Zv[c]].dims[c] for c in xrange(self.n_cols())]

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
            D = get_propsal_dim(self.dims(col), v)
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
            D = get_propsal_dim(self.dims(col), None)
            V = View(self.X, [])
            logp = V.incorporate_dim(D)
            p_view.append(logp + p_crp_aux)
            proposal_dims.append(D)
            proposal_views.append(V)

        # Enforce independence constraints.
        avoid = [a for p in self.Ci if col in p for a in p if a != col]
        for a in avoid:
            p_view[self.Zv[a]] = float('-inf')

        # Draw view.
        v_b = gu.log_pflip(p_view)
        D = proposal_dims[v_b]

        # Append auxiliary view?
        if v_b >= len(self.views):
            self.views.append(proposal_views[v_b-len(self.Nv)])
            self.Nv.append(0)
            v_b = len(self.Nv)-1

        # Accounting.
        if v_a != v_b:
            self.views[v_a].unincorporate_dim(D)
        self.views[v_b].incorporate_dim(D, reassign=D.is_collapsed())
        self.Zv[col] = v_b
        self.Nv[v_a] -= 1
        self.Nv[v_b] += 1

        # Delete empty view?
        if self.Nv[v_a] == 0:
            self.Zv = [i-1 if i>v_a else i for i in self.Zv]
            del self.Nv[v_a]
            del self.views[v_a]

        self._check_partitions()

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
        assert len(self.dims()) == self.n_cols()
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
        # Dependence constraints.
        assert vu.validate_crp_constrained_partition(self.Zv, self.Cd, self.Ci)

    # --------------------------------------------------------------------------
    # Serialize

    def to_metadata(self):
        metadata = dict()

        # Dataset.
        metadata['X'] = self.X.tolist()

        # Misc data.
        metadata['seed'] = self.seed

        # View partition data.
        metadata['alpha'] = self.alpha
        metadata['Nv'] = self.Nv
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
        return cls(X, metadata['cctypes'], metadata['distargs'],
            Zv=metadata['Zv'], Zrv=metadata['Zrv'], alpha=metadata['alpha'],
            view_alphas=metadata['view_alphas'], hypers=metadata['hypers'],
            seed=metadata['seed'])

    @classmethod
    def from_pickle(cls, fileptr):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)
