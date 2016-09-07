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
import time

import numpy as np

import bayeslite

from bayeslite.read_csv import bayesdb_read_csv
from bdbcontrib.bql_utils import nullify

from crosscat.LocalEngine import LocalEngine

from cgpm.crosscat import lovecat
from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu

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
bdb.execute('ANALYZE data_m FOR 2 ITERATION WAIT;')

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
    distargs=distargs_prime,
    Zv={o:0 for o in outputs_prime},
    rng=rng)


# Assert that M_c_prime agrees with CrossCat M_c.
M_c_prime = lovecat._crosscat_M_c(state)
M_c = metamodel._crosscat_metadata(bdb, 1)

assert M_c['name_to_idx'] == M_c_prime['name_to_idx']
assert M_c['idx_to_name'] == M_c_prime['idx_to_name']
assert M_c['column_metadata'] == M_c_prime['column_metadata']

# XXX Data

bdb_data = metamodel._crosscat_data(bdb, 1, M_c)
cgpm_data = lovecat._crosscat_T(state, M_c_prime)

assert np.all(np.isclose(bdb_data, cgpm_data, atol=1e-1, equal_nan=True))

# XXX X_L and X_D

theta = metamodel._crosscat_theta(bdb, 1, 0)

X_D = lovecat._crosscat_X_D(state, M_c_prime)
X_L = lovecat._crosscat_X_L(state, M_c_prime, X_D)

LE = LocalEngine(seed=4)
import time
start = time.time()
X_L_new, X_D_new = LE.analyze(
    M_c_prime, lovecat._crosscat_T(state, M_c_prime),
    X_L, X_D, 1, max_time=10, n_steps=150000, max_iterations=150000)
print time.time() - start
lovecat._update_state(state, M_c, X_L_new, X_D_new)

start = time.time()
lovecat.transition_lovecat(state, S=10, seed=None)
print time.time() - start
