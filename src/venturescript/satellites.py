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

import math

import pandas as pd

from vscgpm import VsCGpm


indices = {
    'cluster_id':   0,
    'error':        1,
    'period':       2,
    'apogee':       3,
    'perigee':      4,
    }

satellites = pd.read_csv('../../examples/satellites/satellites.csv')

kepler = VsCGpm(
    outputs=[
        indices['cluster_id'],
        indices['error'],
        indices['period']],
    inputs=[
        indices['apogee'],
        indices['perigee']],
    source='toy.vnt')


for rowid, row in satellites.iterrows():
    A, P, T = row['Apogee_km'], row['Perigee_km'], row['Period_minutes']
    if any(math.isnan(v) for v in [A, P, T]):
        print 'Skipping: %s' % (str((rowid, A, P, T)))
    else:
        query = {indices['period']: T}
        evidence = {indices['apogee']: A, indices['perigee']: P}
        kepler.incorporate(rowid, query, evidence)
    if rowid > 25:
        break
