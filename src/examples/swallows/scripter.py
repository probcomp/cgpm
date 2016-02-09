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
import numpy as np
from gpmcc.engine import Engine

# Load the data.
filenames = [
    'resources/20160209-080359-swallows.engine',
    'resources/20160209-081620-swallows.engine',
    'resources/20160209-083211-swallows.engine',
    'resources/20160209-083710-swallows.engine']

engines = []
names = []
for f in filenames:
    with open(f,'r') as infile:
        engines.append(Engine.from_pickle(infile))
        names.append(engines[-1].get_state(0).cctypes()[1])
T = engines[0].get_state(0).X

# Histogram the data.
N = 100
width = (2*np.pi) / N

fig = plt.figure()
fig.add_subplot(211, projection='polar')
fig.add_subplot(212, projection='polar')

fig.axes[0].set_yticklabels([])
fig.axes[0].hist(
    T[T[:,0]==0][:,1]*np.pi/180., width=width, bottom=1, alpha=.4,
    color='r', bins=100, normed=1)

fig.axes[1].set_yticklabels([])
fig.axes[1].hist(
    T[T[:,0]==1][:,1]*np.pi/180., width=width, bottom=1, alpha=.4,
    color='g',bins=100, normed=1)

# Boxplot the marginals.
marginals = [e.logpdf_marginal() for e in engines]
# XXX Adjust the beta pdf for scaling by 360.
scale = 360
marginals[names.index('beta_uc')] = [
    m - len(T)*np.log(scale) for m in marginals[names.index('beta_uc')]]

lowest_marginals = [np.argmin(l) for l in marginals]
states_marginals = [e.get_state(i) for e,i in zip(engines,lowest_marginals)]

_, ax = plt.subplots()
ax.boxplot(marginals, labels=names)

# Violinplot the marginals.
colors = ['r','b','g','y']
_, ax = plt.subplots()
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(names)
ax.set_xlim([0,5])
vp = ax.violinplot(marginals)
for pc, c in zip(vp['bodies'], colors):
    pc.set_facecolor(c)
vp['cbars'].set_color('k')
vp['cmins'].set_color('k')
vp['cmaxes'].set_color('k')
vp['cbars'].set_alpha(.3)
vp['cmins'].set_alpha(.3)
vp['cmaxes'].set_alpha(.3)

# Obtain the cluster counts from each engine.
cluster_counts = [
    [len(set(e.metadata[i]['Zrcv'][0])) for i in xrange(e.num_states)]
    for e in engines]
lowest_counts = [np.argmin(l) for l in cluster_counts]
states_counts = [e.get_state(i) for e,i in zip(engines, lowest_counts)]

# XXX Make the normal_trunc look like normal with 2 clusters, and make the arg
# that we have better model by reassigning weight to the interval.
# states_counts[-1].transition(kernels=['rows'],N=111)

def logmeanexp(array):
    from scipy.misc import logsumexp
    return logsumexp(array) - np.log(len(array))

# Compute predictive densities.
grid = 100
rowids = [(-1,)] * grid
evidences = [[] for _ in xrange(len(rowids))]

# Compute normal predictive.
e = engines[names.index('normal')]
queries = [[(1,i)] for i in np.linspace(0.01, 359.8, grid)]
L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
logpdfs_normal = [logmeanexp(L[:,i]) for i in xrange(grid)]

# Compute normaltrunc predictive.
e = engines[names.index('normal_trunc')]
queries = [[(1,i)] for i in np.linspace(0.01, 359.8, grid)]
L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
logpdfs_trunc = [logmeanexp(L[:,i]) for i in xrange(grid)]

# Compute vonmises predictive.
e = engines[names.index('vonmises')]
queries = [[(1,i)] for i in np.linspace(0.01, 2*np.pi-0.01, grid)]
L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
logpdfs_vonmises = [logmeanexp(L[:,i]) for i in xrange(grid)]

# Compute beta_uc predictive.
e = engines[names.index('beta_uc')]
queries = [[(1,i)] for i in np.linspace(0.01, 0.99, grid)]
L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
logpdfs_beta = [logmeanexp(L[:,i]-np.log(scale)) for i in xrange(grid)]
