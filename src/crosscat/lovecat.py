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

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
from past.utils import old_div
import sys

import numpy as np

from cgpm.mixtures.view import View
from cgpm.utils import timer as tu


def _crosscat_M_c(state):
    """Create M_c from cgpm.state.State"""
    T = state.X
    outputs = state.outputs
    cctypes = state.cctypes()
    distargs = state.distargs()

    assert len(T) == len(outputs) == len(cctypes) == len(distargs)
    assert all(c in ['normal', 'categorical'] for c in cctypes)
    ncols = len(outputs)

    def create_metadata_numerical():
        return {
            str('modeltype'): str('normal_inverse_gamma'),
            str('value_to_code'): {},
            str('code_to_value'): {},
        }
    def create_metadata_categorical(col, k):
        categories = [v for v in sorted(set(T[col])) if not np.isnan(v)]
        assert all(0 <= c < k for c in categories)
        codes = [str('%d') % (c,) for c in categories]
        ncodes = list(range(len(codes)))
        return {
            str('modeltype'):
                str('symmetric_dirichlet_discrete'),
            str('value_to_code'):
                dict(list(zip(list(map(str, ncodes)), codes))),
            str('code_to_value'):
                dict(list(zip(codes, ncodes))),
        }

    column_names = [str('c%d') % (i,) for i in outputs]
    # Convert all numerical datatypes to normal for lovecat.
    column_metadata = [
        create_metadata_numerical() if cctype != 'categorical' else\
            create_metadata_categorical(output, distarg['k'])
        for output, cctype, distarg in zip(outputs, cctypes, distargs)
    ]

    return {
        str('name_to_idx'):
            dict(list(zip(column_names, list(range(ncols))))),
        str('idx_to_name'):
            dict(list(zip(list(map(str, list(range(ncols)))), column_names))),
        str('column_metadata'):
            column_metadata,
    }

def _crosscat_T(state, M_c):
    """Create T from cgpm.state.State"""
    T = state.X
    def crosscat_value_to_code(val, col):
        if np.isnan(val):
            return val
        # For hysterical raisins, code_to_value and value_to_code are
        # backwards, so to convert from a raw value to a crosscat value we
        # need to do code->value.
        lookup = M_c['column_metadata'][col]['code_to_value']
        if lookup:
            assert str(int(val)) in lookup
            return float(lookup[str(int(val))])
        else:
            return val
    ordering = state.outputs
    rows = list(range(len(T[ordering[0]])))
    return [
        [crosscat_value_to_code(T[col][row], i)
            for (i, col) in enumerate(ordering)]
        for row in rows
    ]


def _crosscat_X_D(state, M_c):
    """Create X_D from cgpm.state.State"""
    view_assignments = list(state.Zv().values())
    views_unique = sorted(set(view_assignments))

    cluster_assignments = [
        list(state.views[v].Zr().values())
        for v in views_unique
    ]
    cluster_assignments_unique = [
        sorted(set(assgn))
        for assgn in cluster_assignments
    ]
    cluster_assignments_to_code = [
        {k:i for (i,k) in enumerate(assgn)}
        for assgn in cluster_assignments_unique
    ]
    cluster_assignments_remapped = [
        [coder[v] for v in assgn] for (coder, assgn)
        in zip(cluster_assignments_to_code, cluster_assignments)
    ]
    # cluster_assignments_remapped[i] contains the row partition for the
    # views_unique[i].
    return cluster_assignments_remapped


def _crosscat_X_L(state, M_c, X_D):
    """Create X_L from cgpm.state.State"""

    # -- Generates X_L['column_hypers'] --
    def column_hypers_numerical(index, hypers):
        assert state.cctypes()[index] != 'categorical'
        return {
            str('fixed'): 0.0,
            str('mu'): hypers['m'],
            str('nu'): hypers['nu'],
            str('r'): hypers['r'],
            str('s'): hypers['s'],
        }

    def column_hypers_categorical(index, hypers):
        assert state.cctypes()[index] == 'categorical'
        K = len(M_c['column_metadata'][index]['code_to_value'])
        assert K > 0
        return {
            str('fixed'): 0.0,
            str('dirichlet_alpha'): hypers['alpha'],
            str('K'): K
        }

    # Retrieve the column_hypers.
    column_hypers = [
        column_hypers_numerical(i, state.dims()[i].hypers)
            if cctype != 'categorical'
            else column_hypers_categorical(i, state.dims()[i].hypers)
        for i, cctype in enumerate(state.cctypes())
    ]

    # -- Generates X_L['column_partition'] --
    view_assignments = list(state.Zv().values())
    views_unique = sorted(set(view_assignments))
    views_to_code = {v:i for (i,v) in enumerate(views_unique)}
    # views_remapped[i] contains the zero-based view index for
    # state.outputs[i].
    views_remapped = [views_to_code[state.Zv(o)] for o in state.outputs]
    counts = list(np.bincount(views_remapped))
    assert 0 not in counts
    column_partition = {
        str('assignments'): views_remapped,
        str('counts'): counts,
        str('hypers'): {str('alpha'): state.alpha()}
    }

    # -- Generates X_L['view_state'] --
    def view_state(v):
        view = state.views[v]
        row_partition = X_D[views_to_code[v]]
        # Generate X_L['view_state'][v]['column_component_suffstats']
        numcategories = len(set(row_partition))
        column_component_suffstats = [
            [{} for c in range(numcategories)]
            for d in view.dims]

        # Generate X_L['view_state'][v]['column_names']
        column_names = \
            [str('c%d' % (o,)) for o in view.outputs[1:]]

        # Generate X_L['view_state'][v]['row_partition_model']
        counts = list(np.bincount(row_partition))
        assert 0 not in counts

        return {
            str('column_component_suffstats'):
                column_component_suffstats,
            str('column_names'):
                column_names,
            str('row_partition_model'): {
                str('counts'): counts,
                str('hypers'): {str('alpha'): view.alpha()}
            }
        }

    # view_states[i] is the view for code views_to_code[i], so we need to
    # iterate in the same order of views_unique to agree with both X_D (the row
    # partition in each view), as well as X_L['column_partition']['assignments']
    view_states = [view_state(v) for v in views_unique]

    # Generates X_L['col_ensure'].
    col_ensure = dict()
    if state.Cd:
        col_ensure['dependent'] = {
            str(column) : list(block)
            for block in state.Cd for column in block
        }
    if state.Ci:
        from crosscat.utils.general_utils import get_scc_from_tuples
        col_ensure['independent'] = {
            str(column) : list(block) for
            column, block in get_scc_from_tuples(state.Ci).items()
        }

    return {
        str('column_hypers'): column_hypers,
        str('column_partition'): column_partition,
        str('view_state'): view_states,
        str('col_ensure'): col_ensure
    }


def _update_state(state, M_c, X_L, X_D):
    # Perform checking on M_c.
    assert all(c in ['normal','categorical'] for c in state.cctypes())
    assert len(M_c['name_to_idx']) == len(state.outputs)
    def _check_model_type(i):
        reference = 'normal_inverse_gamma' if state.cctypes()[i] == 'normal'\
            else 'symmetric_dirichlet_discrete'
        return M_c['column_metadata'][i]['modeltype'] == reference
    assert all(_check_model_type(i) for i in range(len(state.cctypes())))
    # Perform checking on X_D.
    assert all(len(partition)==state.n_rows() for partition in X_D)
    assert len(X_D) == len(X_L['view_state'])
    # Perform checking on X_L.
    assert len(X_L['column_partition']['assignments']) == len(state.outputs)

    # Update the global state alpha.
    state.crp.set_hypers(
        {'alpha': X_L['column_partition']['hypers']['alpha']}
    )

    assert state.alpha() == X_L['column_partition']['hypers']['alpha']
    assert state.crp.clusters[0].alpha ==\
        X_L['column_partition']['hypers']['alpha']

    # Create the new views.
    offset = max(state.views) + 1
    new_views = []
    for v in range(len(X_D)):
        alpha = X_L['view_state'][v]['row_partition_model']['hypers']['alpha']
        index = v + offset

        assert index not in state.views
        view = View(
            state.X,
            outputs=[state.crp_id_view + index],
            Zr=X_D[v],
            alpha=alpha,
            rng=state.rng
        )
        new_views.append(view)
        state._append_view(view, index)

    # Migrate the dims to their view partitions.
    for i, c in enumerate(state.outputs):
        v_a = state.Zv(c)
        v_b = X_L['column_partition']['assignments'][i] + offset
        state._migrate_dim(v_a, v_b, state.dim_for(c))

    # Update the dim hyperparameters.
    # This code is disabled because lovecat may give hypers which result in
    # math domain errors!
    # for i, c in enumerate(state.outputs):
    #     dim = state.dim_for(c)
    #     if dim.cctype == 'categorical':
    #         dim.hypers['alpha'] = X_L['column_hypers'][i]['dirichlet_alpha']
    #     elif dim.cctype == 'normal':
    #         dim.hypers['m'] = X_L['column_hypers'][i]['mu']
    #         dim.hypers['r'] = X_L['column_hypers'][i]['r']
    #         dim.hypers['s'] = X_L['column_hypers'][i]['s']
    #         dim.hypers['nu'] = X_L['column_hypers'][i]['nu']
    #     else:
    #         assert False

    assert len(state.views) == len(new_views)
    state._check_partitions()


def _update_diagnostics(state, diagnostics):
    # Update logscore.
    cc_logscore = diagnostics.get('logscore', np.array([]))
    new_logscore = list(map(float, np.ravel(cc_logscore).tolist()))
    state.diagnostics['logscore'].extend(new_logscore)

    # Update column_crp_alpha.
    cc_column_crp_alpha = diagnostics.get('column_crp_alpha', [])
    new_column_crp_alpha = list(map(float, np.ravel(cc_column_crp_alpha).tolist()))
    state.diagnostics['column_crp_alpha'].extend(list(new_column_crp_alpha))

    # Update column_partition.
    def convert_column_partition(assignments):
        return [
            (col, int(assgn))
            for col, assgn in zip(state.outputs, assignments)
        ]
    new_column_partition = diagnostics.get('column_partition_assignments', [])
    if len(new_column_partition) > 0:
        assert len(new_column_partition) == len(state.outputs)
        trajectories = np.transpose(new_column_partition)[0].tolist()
        state.diagnostics['column_partition'].extend(
            list(map(convert_column_partition, trajectories)))


def _progress(n_steps, max_time, step_idx, elapsed_secs, end=None):
    if end:
        print('\rCompleted: %d iterations in %f seconds.' %\
            (step_idx, elapsed_secs))
    else:
        p_seconds = old_div(elapsed_secs, max_time) if max_time != -1 else 0
        p_iters = float(step_idx) / n_steps
        percentage = max(p_iters, p_seconds)
        tu.progress(percentage, sys.stdout)


def transition(
        state, N=None, S=None, kernels=None, rowids=None, cols=None,
        seed=None, checkpoint=None, progress=None):
    """Runs full Gibbs sweeps of all kernels on the cgpm.state.State object.

    Permittable kernels:
       'column_partition_hyperparameter'
       'column_partition_assignments'
       'column_hyperparameters'
       'row_partition_hyperparameters'
       'row_partition_assignments'
    """

    if seed is None:
        seed = 1
    if kernels is None:
        kernels = ()
    if (progress is None) or progress:
        progress = _progress

    if N is None and S is None:
        n_steps = 1
        max_time = -1
    if N is not None and S is None:
        n_steps = N
        max_time = -1
    elif S is not None and N is None:
        # This is a hack, lovecat has no way to specify just max_seconds.
        n_steps = 150000
        max_time = S
    elif S is not None and N is not None:
        n_steps = N
        max_time = S
    else:
        assert False

    if cols is None:
        cols = ()
    else:
        cols = [state.outputs.index(i) for i in cols]
    if rowids is None:
        rowids = ()

    M_c = _crosscat_M_c(state)
    T = _crosscat_T(state, M_c)
    X_D = _crosscat_X_D(state, M_c)
    X_L = _crosscat_X_L(state, M_c, X_D)

    from crosscat.LocalEngine import LocalEngine
    LE = LocalEngine(seed=seed)

    if checkpoint is None:
        X_L_new, X_D_new = LE.analyze(
            M_c, T, X_L, X_D, seed,
            kernel_list=kernels, n_steps=n_steps, max_time=max_time,
            c=cols, r=rowids, progress=progress)
        diagnostics_new = dict()
    else:
        X_L_new, X_D_new, diagnostics_new = LE.analyze(
            M_c, T, X_L, X_D, seed,
            kernel_list=kernels, n_steps=n_steps, max_time=max_time,
            c=cols, r=rowids, do_diagnostics=True,
            diagnostics_every_N=checkpoint, progress=progress)

    _update_state(state, M_c, X_L_new, X_D_new)

    if diagnostics_new:
        _update_diagnostics(state, diagnostics_new)
