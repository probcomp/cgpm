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

switch = 3
dist = ['normal','normal_trunc','vonmises','beta_uc'][switch]

print 'Learning distribution %s' % dist

print 'Loading dataset ...'
df = pd.read_csv('swallows.csv')
df.replace('NA', np.nan, inplace=True)
df.replace('NaN', np.nan, inplace=True)
schema = [('treatment', 'bernoulli'), ('heading', dist)]

print 'Parsing schema ...'
T, cctypes, distargs, valmap, columns = du.parse_schema(schema, df)

# Convert to radians.
T = T.astype(float)
T[:,1] = T[:,1]*np.pi/180

if dist == 'normal_trunc':
    distargs[1] = {'l':0, 'h':2*np.pi}
elif dist == 'beta_uc':
    T[:,1] = T[:,1] / (2*np.pi) + 1e-3

print 'Initializing engine ...'
engine = Engine(T, cctypes, distargs=distargs, num_states=28, initialize=1)

print 'Analyzing for 1000 iterations ...'
engine.transition(N=1, multithread=1)

print 'Pickling ...'
engine.to_pickle(file('%s-swallows.engine' % cu.timestamp(), 'w'))
