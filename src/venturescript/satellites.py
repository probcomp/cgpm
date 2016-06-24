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
import cPickle

import numpy as np
import pandas as pd

from vscgpm import VsCGpm
from cgpm.utils import config as cu


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
        print 'Incorporating: %s' % (str((rowid, A, P, T)))
        query = {indices['period']: T}
        evidence = {indices['apogee']: A, indices['perigee']: P}
        kepler.incorporate(rowid, query, evidence)

if False:
    for rowid, row in satellites.iterrows():
        A, P, T = row['Apogee_km'], row['Perigee_km'], row['Period_minutes']
        if any(math.isnan(v) for v in [A, P, T]):
            print 'Skipping: %s' % (str((rowid, A, P, T)))
        else:
            print 'Checking: %s' % (str((rowid, A, P, T)))
            sample = kepler.simulate(rowid, [indices['period']])
            assert np.allclose(sample[indices['period']], T)

kepler.transition(steps=25000)

clusters = [kepler.simulate(r, [indices['cluster_id']])
    for r in kepler.obs.keys()]
errors = [kepler.simulate(r, [indices['error']])
    for r in kepler.obs.keys()]
X = np.asarray([(a[indices['cluster_id']], b[indices['error']])
    for a,b in zip(clusters, errors)])

timestamp = cu.timestamp()
kepler.ripl.save('resources/%s-kepler.ripl' % timestamp)
cPickle.dump(clusters, file('resources/%s-clusters' % timestamp, 'w'))
cPickle.dump(errors, file('resources/%s-errors' % timestamp, 'w'))
np.save(file('resources/%s-clusters_errors' % timestamp, 'w'), X)
