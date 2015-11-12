import numpy
import random
import pylab
from scipy.stats import gamma
from numpy.random import gamma as gamrnd

from math import log
import gpmcc.utils.general as utils
import gpmcc.utils.sampling as su

class cc_view(object):
    """
    cc_view (CrossCat view)
    """
    def __init__(self, dims, alpha=None, Z=None, n_grid=30):
        """
        Constructor
        input arguments:
        -- dims: a list of cc_dim objects
        optional arguments:
        -- alpha: crp concentration parameter. If none, is selected from grid.
        -- Z: starting partiton of rows to categories. If nonde, is intialized
        from CRP(alpha)
        -- n_grid: number of grid points in the hyperparameter grids
        """

        N = dims[0].N
        self.N = N

        # generate alpha
        self.alpha_grid = utils.log_linspace(1.0/self.N, self.N, n_grid)

        if alpha is None:
            alpha = random.choice(self.alpha_grid)
        else:
            assert alpha > 0.0

        if Z is None:
            Z, Nk, K = utils.crp_gen(N, alpha)
        else:
            assert len(Z) == dims[0].X.shape[0]
            Nk = utils.bincount(Z)
            K = len(Nk)

        assert sum(Nk) == N
        assert K == len(Nk)

        self.dims = dict()
        for dim in dims:
            dim.reassign(Z)
            self.dims[dim.index] = dim

        self.alpha = alpha
        self.Z = numpy.array(Z)
        self.K = K
        self.Nk = Nk

    def reassign_rows_to_cats(self, which_rows=None):
        """
        It do what it say--reassign rows to categories.
        optional arguments:
        -- which_rows: a list of rows to reassign. If not specified, reassigns 
        every row
        """

        log_alpha = log(self.alpha)

        if which_rows is None:
            which_rows = [i for i in range(self.N)]

        # random.shuffle(which_rows)

        for row in which_rows:
            # get the current assignment, z_a, and determine if it is a singleton
            z_a = self.Z[row]
            is_singleton = (self.Nk[z_a] == 1)

            # get CRP probabilities
            pv = list(self.Nk)
            if is_singleton:
                # If z_a is a singleton, do not consider a new singleton
                pv[z_a] = self.alpha
            else:
                pv[z_a] -= 1

            # take the log of the CRP probabilities
            pv = numpy.log(numpy.array(pv))

            ps = []
            # calculate the probability of each row in each category, k \in K
            for k in range(self.K):
                if k == z_a and is_singleton:
                    lp = self.singleton_predictive_logp(row) + pv[k]
                else:
                    lp = self.row_predictive_logp(row,k) + pv[k]
                ps.append(lp)

            # propose singleton
            if not is_singleton:
                lp = self.singleton_predictive_logp(row) + log_alpha
                ps.append(lp)

            # Draw new assignment, z_b
            z_b = utils.log_pflip(ps)

            if z_a != z_b:
                if is_singleton:
                    self.destroy_singleton_cluster(row, z_a, z_b)
                elif z_b == self.K:
                    self.create_singleton_cluster(row, z_a)
                else:
                    self.move_row_to_cluster(row, z_a, z_b)

            # make sure the reassign worked properly
            # assert sum(self.Nk) == self.N
            # assert len(self.Nk) == self.K

            # zs = list(set(self.Z))
            # for j in range(self.K):
            #   assert zs[j] == j
            #   for dim in self.dims.keys():
            #       assert self.dims[dim].clusters[j].N == self.Nk[j]


    def transition(self, N, do_plot=False):
        """
        Do all the transitions. Do_plot is mainly for debugging and is only
        meant to be used to watch multiple transitions in a single view and not
        in full state transitions---cc_state.transition has its own do_plot arg. 
        """
        for _ in range(N):
            self.transition_Z(do_plot)
            self.transition_alpha()
            self.transition_column_hypers()

    def transition_alpha(self):
        """
        Calculate CRP alpha conditionals over grid and transition
        """
        logps = numpy.zeros(len(self.alpha_grid))
        for i in range(len(self.alpha_grid)):
            alpha = self.alpha_grid[i]
            logps[i] = utils.unorm_lcrp_post(alpha, self.N, self.K, lambda x: 0)
        # log_pdf_lambda = lambda a : utils.lcrp(self.n_cols, self.Nv, a) + self.alpha_prior_lambda(a)

        index = utils.log_pflip(logps)
        self.alpha = self.alpha_grid[index]

    def transition_column_hypers(self):
        """
        Calculate column (cc_dim) hyperparameter conditionals over grid and transition
        """
        for dim in self.dims.values():
            dim.update_hypers()

    def transition_Z(self, which_rows=None, N=1):
        """
        Transition row assignment.
        optional arguments:
        -- which_rows: a list of rows to reassign. If not specified, reassigns 
        every row
        -- N: number of times to transition (defualt: 1)
        -- do_plot: plot predictive distribution and data histogram of each 
        dim. For debugging.
        """
        for _ in range(N):
            self.reassign_rows_to_cats(which_rows=which_rows)

    def row_predictive_logp(self, row, cluster):
        """
        Get the predictive log_p of row being in cluster
        """
        z_0 = self.Z[row]   # current assignment
        z_1 = cluster       # queried assignment

        if z_0 == z_1:
            # if the original assignment is the same as the queried cluster, we
            # must remove the data from the suffstats before calculating the 
            # predictive probability
            lp = 0
            for dim in self.dims.values():
                dim.remove_element(row,cluster)         # remove
                lp += dim.predictive_logp(row,cluster)  # calculate
                dim.insert_element(row,cluster)         # reinsert
        else:
            lp = 0
            for dim in self.dims.values():
                lp += dim.predictive_logp(row, cluster)

        return lp

    def singleton_predictive_logp(self, row):
        """
        Get the predictive log_p of row being a singleton cluster
        """
        lp = 0
        for dim in self.dims.values():
            lp += dim.singleton_predictive_logp(row)
        return lp

    def destroy_singleton_cluster(self, row, to_destroy, move_to):
        self.Z[row] = move_to
        zminus = numpy.nonzero(self.Z>to_destroy)
        self.Z[zminus] -= 1
        for dim in self.dims.values():
            dim.destroy_singleton_cluster(row, to_destroy, move_to)

        self.Nk[move_to] += 1
        del self.Nk[to_destroy]
        self.K -= 1

    def create_singleton_cluster(self, row, current):
        self.Z[row] = self.K
        self.K += 1
        self.Nk[current] -= 1
        self.Nk.append(1)

        for dim in self.dims.values():
            dim.create_singleton_cluster(row, current)

    def move_row_to_cluster(self, row, move_from, move_to):
        self.Z[row] = move_to
        self.Nk[move_from] -= 1
        self.Nk[move_to] += 1
        for dim in self.dims.values():
            dim.move_to_cluster(row, move_from, move_to)

    def assimilate_dim(self, new_dim, is_uncollapsed=True):
        # resistance is futile
        if not is_uncollapsed:
            new_dim.reassign(self.Z)
        self.dims[new_dim.index] = new_dim

    def release_dim(self, dim_index):
        del self.dims[dim_index]

