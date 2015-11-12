from baxcat.utils import cc_sample_utils as su
from baxcat.cc_types import cc_normal_model
from baxcat import cc_state

import numpy
import random
#                                   ________        
#                                 ||.     . ||
#                                 ||   -    ||
# _|_|_|    _|      _|    _|_|    ||        ||
# _|    _|  _|_|  _|_|  _|    _| /||--------||\
# _|_|_|    _|  _|  _|  _|    _|  ||===   . ||
# _|    _|  _|      _|  _|    _|  || +  o  0||
# _|_|_|    _|      _|    _|_|    ||________||
#                                    |    |
# READ THIS BEFORE USING
# 1. Currently, this module support only normal data
# 2. Construct_state_from_legacy_metadata ignores the suffstats in X_L and
#    instead calculates them from the partition and the data.

class BaxCatEngine(object):
    """Plays nice with CrossCat"""
    def __init__(self):
        # Nothing to do, why not take your pal, Zoidberg, out to a fancy dinner?
        print "(\/) (',,,,') (\/)"

    def initialize(self, M_c, M_r, T, initialization='from_the_prior',
            specified_s_grid=None, specified_mu_grid=None,
            row_initialization=-1, n_chains=1):
        
        # assumes all columns are normal data
        T = numpy.array(T)
        n_rows, n_cols = T.shape

        X = [ T[:,c] for c in range(n_cols) ]
        cctypes = ['normal']*n_cols
        distargs = [None]*n_cols

        # it don't use M_r
        X_L_list = []
        X_D_list = []
        for chain in range(n_chains):
            state = cc_state.cc_state(X, cctypes, distargs)

            if specified_mu_grid is not None:
                if len(specified_mu_grid) > 0:
                    for dim in state.dims:
                        dim.hypers_grids['m'] = numpy.array(specified_mu_grid)
                        dim.hypers['m'] = random.sample(specified_mu_grid, 1)[0]
                        for cluster in dim.clusters:
                            cluster.set_hypers(dim.hypers)

            if specified_s_grid is not None:
                if len(specified_s_grid) > 0:
                    for dim in state.dims:
                        dim.hypers_grids['s'] = numpy.array(specified_s_grid)
                        dim.hypers['s'] = random.sample(specified_s_grid, 1)[0]
                        for cluster in dim.clusters:
                            cluster.set_hypers(dim.hypers)

            _, X_L, X_D = get_legacy_metadata(state)

            X_L_list.append(X_L)
            X_D_list.append(X_D)

        if n_chains == 1:
            X_L_list, X_D_list = X_L_list[0], X_D_list[0]

        return X_L_list, X_D_list

    def analyze(self, M_c, T, X_L, X_D, 
        specified_s_grid=None, specified_mu_grid=None ):

        # is X_L is a list, then we are running multiple chains (states)
        is_multistate = isinstance(X_L, list)

        if is_multistate:
            n_states = len(X_L)
        else:
            X_L = [X_L]
            X_D = [X_D]
            n_states = 1

        X_L_list = []
        X_D_list = []
        for chain in range(n_states):
            
            state = construct_state_from_legacy_metadata(T, M_c,
                     X_L[chain], X_D[chain])
            # check if we need to update the hyperparameter grids
            if specified_mu_grid is not None:
                if len(specified_mu_grid) > 0:
                    for dim in state.dims:
                        dim.hypers_grids['m'] = numpy.array(specified_mu_grid)

            if specified_s_grid is not None:
                if len(specified_s_grid) > 0:
                    for dim in state.dims:
                        dim.hypers_grids['s'] = numpy.array(specified_s_grid)

            state.transition()

            _, X_Li, X_Di = get_legacy_metadata(state)

            X_L_list.append(X_Li)
            X_D_list.append(X_Di)


        if not is_multistate:
            X_L_list, X_D_list = X_L_list[0], X_D_list[0]

        return X_L_list, X_D_list

    def simple_predictive_sample(self, M_c, X_L, X_D, Y, Q, n=1):
        is_multistate = isinstance(X_L, list)

        # gnseed = lambda : random.randrange(200000)
        # return ccsu.simple_predictive_sample(M_c, X_L, X_D, Y, Q, gnseed)

        if is_multistate:
            # this isn't quite right
            # n_states = len(X_L)
            r = random.randrange(len(X_L))
            x = _do_predictive_sample_legacy(M_c, X_L[r], X_D[r], Y, Q)
            # for i in range(1, n_states):
            #     xi = _do_predictive_sample_legacy(M_c, X_L[i], X_D[i], Y, Q, n)
            #     for q in range(len(Q)):
            #         x[q].extend(xi[q])
        else:
            x = _do_predictive_sample_legacy(M_c, X_L, X_D, Y, Q)

        # x = numpy.transpose(numpy.array(x)).tolist()
        return x

def _do_predictive_sample_legacy(M_c, X_L, X_D, Y, Q):
    # Y is not used. No constrained sampling yet
    # state = construct_state_from_legacy_metadata(T, M_c, X_L, X_D)

    samples = []
    for q in Q:
        row = q[0]
        col = q[1]
        model = _get_normal_model_from_legacy_metadata(row, col, X_L, X_D)
        x = model.predictive_draw()
        samples.append(x)

    return [samples]

def _get_normal_model_from_legacy_metadata(row, col, X_L, X_D):
    """ If row is in n_rows, returns a specific component model """
    # get view
    view = X_L['column_partition']['assignments'][col]

    # get cluster
    cluster = X_D[view][row]

    # get suffstats and hypers
    hypers = X_L['column_hypers'][col]
    suffstats = None
    for view_state in X_L['view_state']:
        try:
            idx = view_state['column_names'].index(col)
        except ValueError:
            idx = None

        if idx is not None:    
            suffstats = view_state['column_component_suffstats'][idx][cluster]

    assert suffstats is not None

    model = cc_normal_model.cc_normal(
            N=suffstats['N'],
            sum_x=suffstats['sum_x'], 
            sum_x_sq=suffstats['sum_x_squared'], 
            m=hypers['mu'], 
            r=hypers['r'], 
            s=hypers['s'], 
            nu=hypers['nu'])

    return model
 

def get_legacy_metadata(state_object):
    """
    get_legacy_metadata. Returns M_c, X_L, and X_D for CrossCat
    """
    n_cols = state_object.n_cols
    n_rows = state_object.n_rows
    n_views = state_object.V

    X_D = []
    for v in range(n_views):
        X_D.append(state_object.views[v].Z.tolist())

    X_L = dict()

    X_L['column_partition'] = {
        'hypers' : {'alpha' : state_object.alpha},
        'assignments'       : state_object.Zv.tolist(),
        'counts'            : state_object.Nv
    }

    X_L['column_hypers'] = []
    X_L['view_state'] = []

    for d in range(state_object.n_cols):
        dim = state_object.dims[d]
        hypers = dim.hypers
        X_L['column_hypers'].append( _get_legacy_column_hypers(dim) )

    for view in state_object.views:
        X_L['view_state'].append(  _get_view_state_for_X_L(view) )

    M_c = _gen_M_c(state_object)

    assert len(X_L['column_partition']['assignments']) == state_object.n_cols
    assert len(X_D) == max(X_L['column_partition']['assignments'])+1
    assert len(X_D[0]) == state_object.n_rows

    return M_c, X_L, X_D

def construct_state_from_legacy_metadata(T, M_c, X_L, X_D):
    """
    Generates a state from CrossCat-formated data, T, and metadata
    """
    # ignores suffstats, calculates them manually
    Zv = X_L['column_partition']['assignments']
    Zrcv = [ Z for Z in X_D ]
    T_array = numpy.array(T)

    X = [ T_array[:,col].flatten(1) for col in range(T_array.shape[1]) ]

    cctypes = ['normal']*len(X)
    distargs = [None]*len(X)

    state = cc_state.cc_state(X, cctypes, distargs, Zv=Zv, Zrcv=Zrcv)

    # set Column alpha
    state.alpha = X_L['column_partition']['hypers']['alpha']

    for v in range(state.V):
        view_state = X_L['view_state'][v]
        state.views[v].alpha = view_state['row_partition_model']['hypers']['alpha']
        for index, dim in state.views[v].dims.iteritems():
            # dict_index = view_state.column_names.index(str(index))
            hypers = X_L['column_hypers'][index]
            model_type = M_c['column_metadata'][index]['modeltype']
            _set_dim_hypers_from_legacy(dim, hypers, model_type)

    return state


def _set_dim_hypers_from_legacy(dim_object, hypers, model_type):
    update_hypers = _get_hypers_for_cc_dim_from_legacy[model_type](dim_object, hypers)
    dim_object.hypers = update_hypers
    for cluster in dim_object.clusters:
        cluster.set_hypers(update_hypers)

def _ghccfl_normal(dim_object, hypers):
    update_hypers = {
        'm' : hypers['mu'],
        's' : hypers['s'],
        'r' : hypers['r'],
        'nu' : hypers['nu'],
    }
    return update_hypers
    
def _get_view_state_for_X_L(view_object):
    view_state = dict()
    view_state['row_partition_model'] = {
        'hypers' : {'alpha': view_object.alpha},
        'counts' : view_object.Nk
    }
    view_state['column_names'] = []

    # column_component_suffstats
    ccsf = []
    for dim in view_object.dims.values():
        view_state['column_names'].append( dim.index )
        this_ccsf = []
        for cluster in dim.clusters:
            this_ccsf.append(_get_legacy_suffstats(cluster))

        ccsf.append(this_ccsf)

    view_state['column_component_suffstats'] = ccsf

    return view_state

def _gen_M_c(state_object):
    M_c = dict()
    M_c['name_to_idx'] = dict()
    M_c['idx_to_name'] = dict()
    M_c['column_metadata'] = []
    for col in range(state_object.n_cols):
        cctype = state_object.dims[col].model.cctype
        M_c['name_to_idx'][col] = col
        M_c['idx_to_name'][col] = col
        M_c['column_metadata'].append(
            { 
                'modeltype'    : _cctype_to_legacy_modeltype[cctype],
                'value_to_code': {},
                'code_to_value': {},
            })
    return M_c

def _get_legacy_suffstats(type_object):
    cctype = type_object.cctype
    hypers = _format_suffstats_for_X_L[cctype](type_object)
    return hypers

def _fxlss_normal(dim_object):
    # format for X_L suffstats
    suffstats = {
        'N'             : dim_object.N,
        'sum_x'         : dim_object.sum_x,
        'sum_x_squared' : dim_object.sum_x_sq }
    return suffstats

def _get_legacy_column_hypers(dim_object):
    # format for X_L column_hypers
    cctype = dim_object.model.cctype
    hypers = dim_object.hypers
    return _fromat_hypers_for_X_L[cctype](hypers)


def _fxlh_normal(hypers):
    # format for X_L column_hypers
    hypers_out = {
        'fixed' : False,
        'mu' : hypers['m'],
        's'  : hypers['s'], 
        'r'  : hypers['r'], 
        'nu' : hypers['nu'] }
    return hypers_out

_cctype_to_legacy_modeltype = {
    'normal' : 'normal_inverse_gamma'
}

_format_suffstats_for_X_L = {
    'normal' : _fxlss_normal
}

_fromat_hypers_for_X_L = {
    'normal' : _fxlh_normal,
}

_get_hypers_for_cc_dim_from_legacy = {
    'normal_inverse_gamma' : _ghccfl_normal,
}