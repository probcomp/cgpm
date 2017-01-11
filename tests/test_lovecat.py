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

import pytest
if not pytest.config.getoption('--integration'):
    pytest.skip('specify --integration to run integration tests')

import StringIO
import contextlib
import time

import numpy as np

import bayeslite

from bayeslite.read_csv import bayesdb_read_csv

from crosscat.LocalEngine import LocalEngine

from cgpm.crosscat import lovecat
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
