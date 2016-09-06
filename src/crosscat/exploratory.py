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

import StringIO

import numpy as np

import bayeslite

from bayeslite.read_csv import bayesdb_read_csv
from bdbcontrib.bql_utils import nullify

from crosscat.LocalEngine import LocalEngine

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu
from cgpm.crosscat.state import State

rng = gu.gen_rng(2)

# Set up the data generation, 20 rows by 8 cols, with some missing values.
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

D, Zv, Zc = tu.gen_data_table(
    20, [1], [[.25, .25, .5]], cctypes, distargs,
    [.95]*len(cctypes), rng=gu.gen_rng(2))

# Generate some missing entries.
missing = rng.choice(range(D.shape[1]), size=(D.shape[0], 4), replace=True)
for i, m in enumerate(missing):
    D[i,m] = np.nan

T = np.transpose(D)

# -------- Create a bdb instance with crosscat -------- #
bdb = bayeslite.bayesdb_open()

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
bdb.execute('ANALYZE data_m FOR 10 ITERATION WAIT;')

# Retrieve the CrossCat metamodel instance.
metamodel = bdb.metamodels['crosscat']

# -------- Create a gpmcc instance -------- #

# Remember that c1 is ignored.
outputs_prime = [0,2,3,4,5,6,7]
cctypes_prime = [c if c == 'categorical' else 'normal'
    for i, c in enumerate(cctypes) if i != 1]
distargs_prime = [d for i, d in enumerate(distargs) if i != 1]
T_prime = {c: T[:,c] for c in outputs_prime}

state = State(
    X=np.transpose([D[o] for o in outputs_prime]),
    outputs=outputs_prime,
    cctypes=cctypes_prime,
    distargs=distargs_prime)

# -------- Create a gpmcc instance -------- #

# Function to create M_c from a state.
def create_metadata(state):
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
        categories = filter(lambda v: not np.isnan(v), sorted(set(T[col])))
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
    column_metadata = [
        create_metadata_numerical() if cctype == 'normal' else\
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

# Assert that M_c_prime agrees with CrossCat M_c.
M_c_prime = create_metadata(state)
M_c = metamodel._crosscat_metadata(bdb, 1)

assert M_c['name_to_idx'] == M_c_prime['name_to_idx']
assert M_c['idx_to_name'] == M_c_prime['idx_to_name']
assert M_c['column_metadata'] == M_c_prime['column_metadata']

# XXX Data

def _crosscat_data(state, M_c):
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

bdb_data = metamodel._crosscat_data(bdb, 1, M_c)
cgpm_data = _crosscat_data(state, M_c_prime)

assert np.all(np.isclose(bdb_data, cgpm_data, atol=1e-1, equal_nan=True))
