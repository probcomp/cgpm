# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import cPickle
import math

import numpy as np
import pandas as pd

from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import data as du
from vscgpm import VsCGpm


satellites = pd.read_csv('../../examples/satellites/satellites.csv')
satellites.replace('NA', np.nan, inplace=True)
satellites.replace('NaN', np.nan, inplace=True)

schema = [
    ('Name',                            'ignore',          -1),
    ('Country_of_Operator',             'categorical',      0),
    ('Operator_Owner',                  'categorical',      1),
    ('Users',                           'categorical',      2),
    ('Purpose',                         'categorical',      3),
    ('Class_of_Orbit',                  'categorical',      4),
    ('Type_of_Orbit',                   'categorical',      5),
    ('Perigee_km',                      'normal',           6),
    ('Apogee_km',                       'normal',           7),
    ('Eccentricity',                    'normal',           8),
    ('Period_minutes',                  'ignore',           9),
    ('Launch_Mass_kg',                  'normal',           10),
    ('Dry_Mass_kg',                     'normal',           11),
    ('Power_watts',                     'normal',           12),
    ('Date_of_Launch',                  'normal',           13),
    ('Anticipated_Lifetime',            'normal',           14),
    ('Contractor',                      'categorical',      15),
    ('Country_of_Contractor',           'categorical',      16),
    ('Launch_Site',                     'categorical',      17),
    ('Launch_Vehicle',                  'categorical',      18),
    ('Source_Used_for_Orbital_Data',    'categorical',      19),
    ('longitude_radians_of_geo',        'normal',           20),
    ('Inclination_radians',             'normal',           21),
    ('kepler_cluster_id',               'ignore',           22),
    ('kepler_error',                    'ignore',           23),
]

T, outputs, cctypes, distargs, valmap, columns = du.parse_schema(schema, satellites)
state = State(T[:10], outputs=outputs, cctypes=cctypes, distargs=distargs)
state.transition(N=1)

kepler = VsCGpm(outputs=[22,23,9], inputs=[7,6], source='toy.vnt')

for rowid, row in satellites.iterrows():
    A, P, T = row['Apogee_km'], row['Perigee_km'], row['Period_minutes']
    if any(math.isnan(v) for v in [A, P, T]):
        print 'Skipping: %s' % (str((rowid, A, P, T)))
    else:
        print 'Incorporating: %s' % (str((rowid, A, P, T)))
        kepler.incorporate(rowid, {9: T}, {7:A, 6:P})
    if rowid > 10:
        break

token = state.compose_cgpm(kepler)

for rowid, row in satellites.iterrows():
    if rowid == 9:  # State only has rowids 0 through 9.
        break
    A, P, T = row['Apogee_km'], row['Perigee_km'], row['Period_minutes']
    if any(math.isnan(v) for v in [A, P, T]):
        print 'Skipping: %s' % (str((rowid, A, P, T)))
    else:
        print 'Checking: %s' % (str((rowid, A, P, T)))
        sample_a = state.simulate(rowid, [9])
        sample_b = state.simulate(rowid, [9])
        assert np.allclose(sample_a[9], T)
        assert np.allclose(sample_b[9], T)
