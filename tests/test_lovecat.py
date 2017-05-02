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

"""
This test suite targets cgpm.crosscat.lovecat
"""

import hacks
import pytest
if not pytest.config.getoption('--integration'):
    hacks.skip('specify --integration to run integration tests')


import StringIO
import contextlib
import itertools
import time

import numpy as np

import bayeslite

from bayeslite.read_csv import bayesdb_read_csv

from crosscat.LocalEngine import LocalEngine

from cgpm.crosscat import lovecat
from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


def nullify(bdb, table, null):
    from bayeslite import bql_quote_name
    qt = bql_quote_name(table)
    for v in (r[1] for r in bdb.sql_execute('PRAGMA table_info(%s)' % (qt,))):
        qv = bql_quote_name(v)
        bdb.sql_execute(
            'UPDATE %s SET %s = NULL WHERE %s = ?' % (qt, qv, qv),
            (null,))

# -- Global variables shared by all module functions.
rng = gu.gen_rng(2)

outputs = range(8)
cctypes, distargs = cu.parse_distargs([
    'normal',
    'poisson',
    'bernoulli',
    'categorical(k=8)',
    'lognormal',
    'categorical(k=4)',
    'beta',
    'vonmises'])


def generate_dataset():
    # Set up the data generation, 20 rows by 8 cols, with some missing values.
    D, Zv, Zc = tu.gen_data_table(
        20, [1], [[.25, .25, .5]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(2))

    # Generate some missing entries in D.
    missing = rng.choice(range(D.shape[1]), size=(D.shape[0], 4), replace=True)
    for i, m in enumerate(missing):
        D[i,m] = np.nan

    T = np.transpose(D)
    return T


def generate_dataset_2():
    # Initialize a dataset with four variables where
    # T3 = T2 are in the same view,
    # T0 = -T1 are in the same view.
    rng = gu.gen_rng(1)
    T3 = rng.normal(loc=0, scale=1, size=20)
    T2 = np.copy(T3)
    T1 = np.concatenate((
        rng.normal(loc=10, scale=1, size=10),
        rng.normal(loc=-10, scale=1, size=10),
    ))
    T0 = -T1
    D = np.column_stack((T3, T2, T1, T0))
    return D


# -------- Create a bdb instance with crosscat -------- #
@contextlib.contextmanager
def generate_bdb(T):
    with bayeslite.bayesdb_open(':memory:') as bdb:
        # Convert data into csv format and load it.
        T_header = str.join(',', ['c%d' % (i,) for i in range(T.shape[1])])
        T_data = str.join('\n', [str.join(',', map(str, row)) for row in T])
        f = StringIO.StringIO('%s\n%s' % (T_header, T_data))
        bayesdb_read_csv(bdb, 'data', f, header=True, create=True)
        nullify(bdb, 'data', 'nan')

        # Create a population, ignoring column 1.
        bdb.execute('''
            CREATE POPULATION data_p FOR data WITH SCHEMA(
                IGNORE c1;
                MODEL c0, c2, c4, c6, c7 AS NUMERICAL;
                MODEL c3, c5 AS CATEGORICAL);
        ''')

        # Create a CrossCat metamodel.
        bdb.execute('''
            CREATE METAMODEL data_m FOR data_p USING crosscat(
                c0 NUMERICAL,
                c2 NUMERICAL,
                c4 NUMERICAL,
                c6 NUMERICAL,
                c7 NUMERICAL,

                c3 CATEGORICAL,
                c5 CATEGORICAL);
        ''')

        bdb.execute('INITIALIZE 1 MODEL FOR data_m;')
        bdb.execute('ANALYZE data_m FOR 2 ITERATION WAIT;')
        yield bdb


# -------- Create a cgpm.state crosscat instance -------- #
def generate_state(T):
    # Remember that c1 is ignored.
    outputs_prime = [0,2,3,4,5,6,7]
    cctypes_prime = [c if c == 'categorical' else 'normal'
        for i, c in enumerate(cctypes) if i != 1]
    distargs_prime = [d for i, d in enumerate(distargs) if i != 1]

    state = State(
        X=np.transpose([T[:,o] for o in outputs_prime]),
        outputs=outputs_prime,
        cctypes=cctypes_prime,
        distargs=distargs_prime,
        Zv={o:0 for o in outputs_prime},
        rng=rng)

    return state


def test_cgpm_lovecat_integration():
    """A mix of unit and integration testing for lovecat analysis."""

    T = generate_dataset()

    with generate_bdb(T) as bdb:

        # Retrieve the CrossCat metamodel instance.
        metamodel = bdb.metamodels['crosscat']

        # Retrieve the cgpm.state
        state = generate_state(T)

        # Assert that M_c_prime agrees with CrossCat M_c.
        M_c_prime = lovecat._crosscat_M_c(state)
        M_c = metamodel._crosscat_metadata(bdb, 1)

        assert M_c['name_to_idx'] == M_c_prime['name_to_idx']
        assert M_c['idx_to_name'] == M_c_prime['idx_to_name']
        assert M_c['column_metadata'] == M_c_prime['column_metadata']

        # Check that the converted datasets match.
        bdb_data = metamodel._crosscat_data(bdb, 1, M_c)
        cgpm_data = lovecat._crosscat_T(state, M_c_prime)
        assert np.all(np.isclose(bdb_data, cgpm_data, atol=1e-2, equal_nan=True))

        # X_L and X_D from the CrossCat state. Not sure what tests to write
        # that acccess theta['X_L'] and theta['X_D'] directly.
        theta = metamodel._crosscat_theta(bdb, 1, 0)

        # Retrieve X_D and X_L from the cgpm.state, and check they can be used
        # as arguments to LocalEngine.analyze.
        X_D = lovecat._crosscat_X_D(state, M_c_prime)
        X_L = lovecat._crosscat_X_L(state, M_c_prime, X_D)

        LE = LocalEngine(seed=4)
        start = time.time()
        X_L_new, X_D_new = LE.analyze(
            M_c_prime, lovecat._crosscat_T(state, M_c_prime),
            X_L, X_D, 1, max_time=20, n_steps=100000000,
            progress=lovecat._progress)
        assert np.allclose(time.time() - start, 20, atol=2)

        # This function call updates the cgpm.state internals to
        # match X_L_new, X_D_new. Check it does not destory the cgpm.state and
        # we can still run transitions.
        lovecat._update_state(state, M_c, X_L_new, X_D_new)
        state.transition(S=5)

        # Invoke a lovecat transition directly through the cgpm.state,
        # for 10000 iters with a 5 second timeout.
        start = time.time()
        state.transition_lovecat(S=7, N=100000)
        # Give an extra second for function call overhead.
        assert 7. <= time.time() - start <= 8.

        # Now invoke by iterations only.
        state.transition_lovecat(N=7, progress=False)

        # Make sure we can now run regular cgpm.state transitions again.
        state.transition(S=5)


def test_lovecat_transition_columns():
    """Test transition_lovecat targeting specific rows and columns."""
    D = generate_dataset_2()

    # Create engine and place each variable in its own view.
    engine = Engine(
        D,
        outputs=[3,2,1,0,],
        cctypes=['normal']*D.shape[1],
        Zv={3:0, 2:1, 1:2, 0:3},
        multiprocess=1,
        num_states=4,
        rng=gu.gen_rng(2),
    )

    # Confirm all variables in singleton views.
    for s in engine.states:
        for a, b in itertools.combinations(s.outputs, 2):
            assert s.Zv(a) != s.Zv(b)

    # Store the row partition assignments of each variable.
    Zr_saved = [
        {var : sorted(s.view_for(var).Zr().items()) for var in s.outputs}
        for s in engine.states
    ]

    # Transition only the column hyperparameters of outputs 2 and 3 should not alter
    # the structure of either row or column partitions.
    engine.transition_lovecat(
        N=100,
        cols=[2,3,],
        kernels=['column_hyperparameters'],
        progress=False,
    )

    for i, s in enumerate(engine.states):
        for a, b in itertools.combinations(s.outputs, 2):
            assert s.Zv(a) != s.Zv(b)

        assert all(
            sorted(s.view_for(var).Zr().items()) == Zr_saved[i][var]
            for var in s.outputs
        )

    # Transition variables 2 and 3 should place them in the same view, without
    # altering the view of variables 0 and 1.
    engine.transition_lovecat(N=100, cols=[2,3,])

    for s in engine.states:
        assert s.Zv(3) == s.Zv(2)
        assert s.Zv(1) != s.Zv(0)

    # Transition only row assignments of outputs 0 and 1 should not alter the
    # view partition.
    engine.transition_lovecat(
        N=100,
        cols=[0,1,],
        kernels=['row_partition_assignments'],
        checkpoint=2,
    )

    for s in engine.states:
        assert s.Zv(3) == s.Zv(2)
        assert s.Zv(1) != s.Zv(0)

    # Transition variables 0 and 1 should put them in the same view.
    engine.transition_lovecat(
        N=100,
        cols=[0,1,],
    )

    for s in engine.states:
        assert s.Zv(3) == s.Zv(2)
        assert s.Zv(1) == s.Zv(0)

    # Add a new column 19 which is sum of 2 and 3, and place it in a singleton
    # view.
    T19 = D[:,0] + D[:,1]
    engine.incorporate_dim(T19, [19], cctype='normal', v=125)

    for s in engine.states:
        assert s.Zv(19) == 125

    # Transition all variables except 19 should not influence its view
    # assignment.
    engine.transition_lovecat(
        N=10,
        cols=[3,2,0,1,],
        checkpoint=2,
    )

    for s in engine.states:
        assert all(s.Zv(19) != s.Zv(o) for o in s.outputs if o != 19)

    # Transition only 19 should place it in the same views as 3 and 2.
    engine.transition_lovecat(
        N=100,
        cols=[19],
    )

    for s in engine.states:
        assert s.Zv(19) == s.Zv(3) == s.Zv(2)


def test_lovecat_transition_rows():
    D = generate_dataset_2()

    # Create engine and place each variable in its own view.
    engine = Engine(
        D,
        outputs=[3,2,1,0,],
        cctypes=['normal']*D.shape[1],
        Zv={3:0, 2:1, 1:2, 0:3},
        multiprocess=1,
        num_states=4,
        rng=gu.gen_rng(2),
    )

    # Store the row partition assignments of each variable.
    Zr_saved = [
        {var : sorted(s.view_for(var).Zr().items()) for var in s.outputs}
        for s in engine.states
    ]

    # Transition some rowids 0, 1, 2, 3 only.
    rowids = range(10)
    engine.transition_lovecat(
        N=10,
        rowids=rowids,
        kernels=['row_partition_assignments'],
        progress=False,
    )

    def check_partitions_match(P0, P1):
        c0 = set(p[1] for p in P0)
        c1 = set(p[1] for p in P1)
        blocks0 = set([tuple([p[0] for p in P0 if p[1] == c]) for c in c0])
        blocks1 = set([tuple([p[0] for p in P1 if p[1] == c]) for c in c1])
        return blocks0 == blocks1

    # Check all other rowids have not been changed, and check that at least one
    # rowid from the transitioned ones has changed (not guaranteed, but very
    # likely that there is at least _one_ change).
    all_rowids_match = True
    for i, s in enumerate(engine.states):
        for a, b in itertools.combinations(s.outputs, 2):
            assert s.Zv(a) != s.Zv(b)

        all_rowids_match_s = True

        for var in s.outputs:
            Zr_new = filter(
                lambda (r,c): r not in rowids,
                s.view_for(var).Zr().iteritems()
            )
            Zr_old = filter(
                lambda (r,c): r not in rowids,
                Zr_saved[i][var]
            )
            assert check_partitions_match(Zr_new, Zr_old)

            all_rowids_match_s = (
                all_rowids_match_s and
                check_partitions_match(
                    s.view_for(var).Zr().iteritems(),
                    Zr_saved[i][var],
                ))
        all_rowids_match = \
            all_rowids_match and all_rowids_match_s

    assert not all_rowids_match
