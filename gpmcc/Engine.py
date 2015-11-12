import baxcat.utils.cc_unit_utils as uu
import baxcat.utils.cc_plot_utils as pu
import baxcat.utils.cc_general_utils as gu
import baxcat.utils.cc_validation_utils as vu

from baxcat import cc_state

import scipy.cluster.hierarchy as sch
import multiprocessing
import itertools
import numpy
import scipy
import pylab
import copy

_all_transition_kernels = ['column_z','state_alpha','row_z','column_hypers','view_alphas']

_cctypes_list = ['normal', 'normal_uc','binomial','multinomial',
                 'lognormal','poisson','vonmises','vonmises_uc']


def _do_predictive_sample():
    pass

def _do_predictive_probability():
    pass

def _do_append_feature(args):
    X = args[0]
    X_f = args[1]
    cctype = args[2]
    distargs = args[3]
    ct_kernel = args[4]
    m = args[5]
    metadata = args[6]

    S = cc_state.cc_state.from_metadata(X, metadata=metadata)
    S.append_dim(X_f, cctype, distargs=distargs, ct_kernel=0, m=1)
    
    return S.get_metadata()

def _do_transition(args):
    N = args[0]
    X = args[1]
    kernel_list = args[2]
    ct_kernel = args[3]
    which_rows = args[4]
    which_cols = args[5]
    metadata = args[6]

    S = cc_state.cc_state.from_metadata(X, metadata=metadata)
    S.transition(N, kernel_list, ct_kernel, which_rows, which_cols)

    return S.get_metadata()

def _do_intialize(args):
    X = args[0]
    cctypes = args[1]
    distargs = args[2]
    init_mode = args[3]

    S = cc_state.cc_state(X, cctypes, distargs=distargs)

    return S.get_metadata()

class Engine(object):
    """docstring for Engine"""
    def __init__(self, multithread=True):
        self.multithread = multithread
        if self.multithread:
            self.pool = multiprocessing.Pool()
        self.metadata = None
        pass

    def initialize(self, X, cctypes, distargs, num_states=1, col_names=None, init_mode='from prior'):
        """
        Get or set the initial state
        """
        vu.validate_data(X)
        vu.validate_cctypes(cctypes)

        self.X = X
        self.cctypes = cctypes
        self.num_states = num_states
        self.n_rows = len(X[0])
        self.n_cols = len(X)

        if col_names is None:
            self.col_names = [ "col_" + str(i) for i in range(len(X)) ]
        else:
            assert( isinstance(col_names, list) )
            assert( len(col_names) == len(X) )
            self.col_names = col_names
        args = [(X, cctypes, distargs, init_mode) for _ in range(self.num_states)]

        if self.multithread:
            self.metadata = self.pool.map( _do_intialize, args )
        else:
            self.metadata = [r for r in map( _do_intialize, args )]

    def load_csv(self, filename, cctypes, distargs, num_states, init_mode='from prior'):
        X, col_names = gu.csv_to_data_and_colnames(filename)

        vu.validate_data(X)
        vu.validate_cctypes(cctypes)

        self.X = X
        self.cctypes = cctypes
        self.num_states = num_states
        self.col_names = col_names
        self.X = gu.clean_data(self.X, self.cctypes)

        args = [(self.X, self.cctypes, distargs, init_mode) for _ in range(self.num_states)]

        if self.multithread:
            self.metadata = self.pool.map( _do_intialize, args )
        else:
            self.metadata = [r for r in map( _do_intialize, args )]

    def append_feature(self, X_f, cctype, distargs=None, ct_kernel=0, m=1, col_name=None):
        """
        Add a feature (column) to the data.
        """
        # check data size
        if X_f.shape[0] != self.n_rows:
            raise ValueError("X_f has %i rows; should have %i." % (X_f.shape[0], self.n_rows))

        # check that cc_type is valid
        if cctype not in _cctypes_list:
            errmsg = "%s is an invalid cc_type. Valid types are: %s" % (str(cctype), str(_cctypes_list))
            raise ValueError(errmsg)        

        args = [ (self.X, X_f, cctype, distargs, ct_kernel, m, self.metadata[i]) for i in range(self.num_states) ]

        if self.multithread:
            self.metadata = self.pool.map(_do_append_feature, args)
        else:
            self.metadata = [r for r in map(_do_append_feature, args)]

        self.X.append(X_f)
        self.cctypes.append(cctype)

        if col_name is None:
            self.col_names.append( 'col_' + str(len(self.X)-1) )
        else:
            self.col_names.append( col_name )

    def add_object(self, X_o, update_hypers_grid=False):
        """
        Add an object (row) to the cc_state
        """
        pass

    def transition(self, N=1, kernel_list=None, ct_kernel=2, which_rows=None, which_cols=None):
        """
        Do transitions
        """
        args = [ (N, self.X, kernel_list, ct_kernel, which_rows, which_cols, self.metadata[i]) for i in range(self.num_states) ]

        if self.multithread:
            self.metadata = self.pool.map(_do_transition, args)
        else:
            self.metadata = [r for r in map(_do_transition, args)]

    def predictive_probabiilty(self, query, constraints=None):
        """
        predictive probabiilty
        """

    def predictive_sample(self, query, constraints=None):
        """
        predictive sample
        """

    def impute(self, query):
        """
        Impute data
        """

    def plot_Z(self):
        """
        plot data in different ways
        """
        Zvs = [md['Zv'] for md in self.metadata]
        col_names = list(self.col_names)

        pu.generate_Z_matrix(Zvs, col_names)

    

