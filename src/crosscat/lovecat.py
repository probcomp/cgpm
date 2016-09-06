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

import numpy as np

def _create_metadata(state):
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
            unicode('modeltype'): unicode('normal_inverse_gamma'),
            unicode('value_to_code'): {},
            unicode('code_to_value'): {},
        }
    def create_metadata_categorical(col, k):
        categories = [v for v in sorted(set(T[col])) if not np.isnan(v)]
        assert all(0 <= c < k for c in categories)
        codes = [unicode('%d') % (c,) for c in categories]
        ncodes = range(len(codes))
        return {
            unicode('modeltype'):
                unicode('symmetric_dirichlet_discrete'),
            unicode('value_to_code'):
                dict(zip(map(unicode, ncodes), codes)),
            unicode('code_to_value'):
                dict(zip(codes, ncodes)),
        }

    column_names = [unicode('c%d') % (i,) for i in outputs]
    # Convert all numerical datatypes to normal for lovecat.
    column_metadata = [
        create_metadata_numerical() if cctype != 'categorical' else\
            create_metadata_categorical(output, distarg['k'])
        for output, cctype, distarg in zip(outputs, cctypes, distargs)
    ]

    return {
        unicode('name_to_idx'):
            dict(zip(column_names, range(ncols))),
        unicode('idx_to_name'):
            dict(zip(map(unicode, range(ncols)), column_names)),
        unicode('column_metadata'):
            column_metadata,
    }

def _crosscat_data(state, M_c):
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
            assert unicode(int(val)) in lookup
            return float(lookup[unicode(int(val))])
        else:
            return val
    ordering = sorted(T.keys())
    rows = range(len(T[ordering[0]]))
    return [[crosscat_value_to_code(T[col][row], i) for (i, col) in
        enumerate(ordering)] for row in rows]


def _crosscat_X_D(state, M_c):
    """Create X_D from cgpm.state.State"""
    view_assignments = state.Zv().values()
    views_unique = sorted(set(view_assignments))

    cluster_assignments = [state.views[v].Zr().values() for v in views_unique]
    cluster_assignments_unique = [sorted(set(assgn))
        for assgn in cluster_assignments]
    cluster_assignments_to_code = [{k:i for (i,k) in enumerate(assgn)}
        for assgn in cluster_assignments_unique]
    cluster_assignments_remapped = [
        [coder[v] for v in assgn] for (coder, assgn)
        in zip(cluster_assignments_to_code, cluster_assignments)]

    return cluster_assignments_remapped


def _crosscat_X_L(state, X_D, M_c):
    """Create X_L from cgpm.state.State"""

    # -- Generates X_L['column_hypers'] --
    def column_hypers_numerical(index, hypers):
        assert state.cctypes()[index] != 'categorical'
        return {
            unicode('fixed'): 0.0,
            unicode('mu'): hypers['m'],
            unicode('nu'): hypers['nu'],
            unicode('r'): hypers['r'],
            unicode('s'): hypers['s'],
        }

    def column_hypers_categorical(index, hypers):
        assert state.cctypes()[index] == 'categorical'
        K = len(M_c['column_metadata'][index]['code_to_value'])
        assert K > 0
        return {
            unicode('fixed'): 0.0,
            unicode('dirichlet_alpha'): hypers['alpha'],
            unicode('K'): K
        }

    # Retrieve the column_hypers.
    column_hypers = [
        column_hypers_numerical(i, state.dims()[i].hypers)
            if cctype != 'categorical'
            else column_hypers_categorical(i, state.dims()[i].hypers)
        for i, cctype in enumerate(state.cctypes())
    ]

    # -- Generates X_L['column_partition'] --
    view_assignments = state.Zv().values()
    views_unique = sorted(set(view_assignments))
    views_to_code = {v:i for (i,v) in enumerate(views_unique)}
    views_remapped = [views_to_code[v] for v in view_assignments]
    counts = list(np.bincount(views_remapped))
    assert 0 not in counts
    column_partition = {
        unicode('assignments'): views_remapped,
        unicode('counts'): counts,
        unicode('hypers'): {unicode('alpha'): state.alpha()}
    }

    # -- Generates X_L['view_state'] --
    def view_state(v):
        view = state.views[v]
        row_partition = X_D[views_to_code[v]]
        # Generate X_L['view_state'][v]['column_component_suffstats']
        numcategories = len(set(row_partition))
        column_component_suffstats = [
            [{} for c in xrange(numcategories)]
            for d in view.dims]

        # Generate X_L['view_state'][v]['column_names']
        column_names = \
            [unicode('c%d' % (o,)) for o in state.views[0].outputs[1:]]

        # Generate X_L['view_state'][v]['row_partition_model']
        counts = list(np.bincount(row_partition))
        assert 0 not in counts

        return {
            unicode('column_component_suffstats'):
                column_component_suffstats,
            unicode('column_names'):
                column_names,
            unicode('row_partition_model'): {
                unicode('counts'): counts,
                unicode('hypers'): {unicode('alpha'): view.alpha()}
            }
        }

    view_states = [view_state(v) for v in state.views.keys()]

    return {
        unicode('column_hypers'): column_hypers,
        unicode('column_partition'): column_partition,
        unicode('view_state'): view_states
    }
