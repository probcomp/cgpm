# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import pandas as pd
import numpy as np

import gpmcc.utils.config as cu
import gpmcc.utils.data as du
from gpmcc.engine import Engine

print 'Loading dataset ...'
df = pd.read_csv('satellites.csv')
df.replace('NA', np.nan, inplace=True)
df.replace('NaN', np.nan, inplace=True)

schema = [
    ('Name', 'ignore'),
    ('Country_of_Operator', 'categorical'),
    ('Operator_Owner', 'categorical'),
    ('Users', 'categorical'),
    ('Purpose', 'categorical'),
    ('Class_of_Orbit', 'categorical'),
    ('Type_of_Orbit', 'categorical'),
    ('Perigee_km', 'lognormal'),
    ('Apogee_km', 'lognormal'),
    ('Eccentricity', 'beta_uc'),
    ('Period_minutes', 'lognormal'),
    ('Launch_Mass_kg', 'lognormal'),
    ('Dry_Mass_kg', 'lognormal'),
    ('Power_watts', 'lognormal'),
    ('Date_of_Launch', 'lognormal'),
    ('Anticipated_Lifetime', 'lognormal'),
    ('Contractor', 'categorical'),
    ('Country_of_Contractor', 'categorical'),
    ('Launch_Site', 'categorical'),
    ('Launch_Vehicle', 'categorical'),
    ('Source_Used_for_Orbital_Data', 'categorical'),
    ('longitude_radians_of_geo', 'normal'),
    ('Inclination_radians', 'normal'),
]

print 'Parsing schema ...'
T, cctypes, distargs, valmap, columns = du.parse_schema(schema, df)
T[:,8] += 0.001 # XXX Small hack for beta variables.

print 'Initializing engine ...'
engine = Engine(T, cctypes, distargs=distargs, num_states=48, initialize=1)

print 'Analyzing for 28800 seconds (8 hours) ...'
engine.transition(S=28800, multithread=1)

print 'Pickling ...'
engine.to_pickle(file('%s-satellites.engine' % cu.timestamp(), 'w'))
