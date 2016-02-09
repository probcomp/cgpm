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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gpmcc.utils.config as cu
import gpmcc.utils.data as du
from gpmcc.engine import Engine

print 'Loading dataset ...'
df = pd.read_csv('swallows.csv')
df.replace('NA', np.nan, inplace=True)
df.replace('NaN', np.nan, inplace=True)
schema = [('treatment', 'bernoulli'), ('heading', 'normal_trunc')]

print 'Parsing schema ...'
T, cctypes, distargs, valmap, columns = du.parse_schema(schema, df)

N = 100
bottom = 1
width = (2*np.pi) / N

ax = plt.subplot(211, polar=True)
ax.set_yticklabels([])
bars = ax.hist(
    T[T[:,0]==0][:,1]*np.pi/180., width=width, bottom=bottom, alpha=.4,
    color='r', bins=100, normed=1)

ax = plt.subplot(212, polar=True)
ax.set_yticklabels([])
bars = ax.hist(
    T[T[:,0]==1][:,1]*np.pi/180., width=width, bottom=bottom, alpha=.4,
    color='g',bins=100, normed=1)
