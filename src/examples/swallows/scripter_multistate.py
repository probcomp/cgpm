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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gpmcc.engine import Engine

# Load the train and test sets.
test_sets = []
train_sets = []

for f in os.listdir('splits'):
    (num, kind, _) = f.split('-')
    if kind == 'train':
        train_sets.append(pd.read_csv('splits/%s' % f, index_col=0))
    else:
        test_sets.append(pd.read_csv('splits/%s' % f, index_col=0))

# Number of train/test sets.
num_sets = len(test_sets)

# Load the engines.
filenames = [
    '20160209-223338-normal-0-swallows.engine',
    '20160209-223804-normal-1-swallows.engine',
    '20160209-224224-normal-2-swallows.engine',
    '20160209-224705-normal-3-swallows.engine',
    '20160209-225108-normal-4-swallows.engine',
    '20160209-223228-vonmises-0-swallows.engine',
    '20160209-223546-vonmises-1-swallows.engine',
    '20160209-223909-vonmises-2-swallows.engine',
    '20160209-224159-vonmises-3-swallows.engine',
    '20160209-224539-vonmises-4-swallows.engine',
    '20160209-224209-beta_uc-0-swallows.engine',
    '20160209-225253-beta_uc-1-swallows.engine',
    '20160209-230333-beta_uc-2-swallows.engine',
    '20160209-231404-beta_uc-3-swallows.engine',
    '20160209-232452-beta_uc-4-swallows.engine',
    '20160209-225353-normal_trunc-0-swallows.engine',
    '20160209-233249-normal_trunc-2-swallows.engine',
    '20160209-235254-normal_trunc-3-swallows.engine',
    '20160210-001307-normal_trunc-4-swallows.engine',
    '20160209-231324-normal_trunc-1-swallows.engine',
    ]

# Store in memory.
engines = [ [None for _ in xrange(5)] for _ in xrange(4)]
names = ['normal', 'vonmises', 'beta_uc', 'normal_trunc']
for f in filenames:
    with open('resources/%s' % f,'r') as infile:
        engine = Engine.from_pickle(infile)
        (_, _, dist, num, _) = f.split('-')
        engines[names.index(dist)][int(num)] = engine

# Boxplot the marginals.
marginals = np.asarray(
    [[e.logpdf_marginal() for e in engines[i]] for i in xrange(4)])

# lowest_marginals = [np.argmin(l) for l in marginals]
# states_marginals = [e.get_state(i) for e,i in zip(engines,lowest_marginals)]

def logmeanexp(array):
    from scipy.misc import logsumexp
    return logsumexp(array) - np.log(len(array))

for i in xrange(num_sets):
    # _, ax = plt.subplots()
    # ax.boxplot(marginals[:,i,:].T, labels=names)

    # Violinplot the marginals.
    colors = ['r','b','g','y']
    _, ax = plt.subplots()
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(names)
    ax.set_xlim([0,5])
    vp = ax.violinplot(marginals[:,i,:].T)
    for pc, c in zip(vp['bodies'], colors):
        pc.set_facecolor(c)
    vp['cbars'].set_color('k')
    vp['cmins'].set_color('k')
    vp['cmaxes'].set_color('k')
    vp['cbars'].set_alpha(.3)
    vp['cmins'].set_alpha(.3)
    vp['cmaxes'].set_alpha(.3)

    # Compute the predictives.
    T = test_sets[0]
    rowids = [-1] * len(T)
    queries = []
    for i, row in enumerate(T.iterrows()):
        treatment, heading = row[1][0], row[1][1]
        treatment = 0 if treatment == 'control' else 1
        heading *= np.pi/180
        queries.append([(0, treatment), (1, heading)])

# # Obtain the cluster counts from each engine.
# cluster_counts = [
#     [len(set(e.metadata[i]['Zrcv'][0])) for i in xrange(e.num_states)]
#     for e in engines]
# lowest_counts = [np.argmin(l) for l in cluster_counts]
# states_counts = [e.get_state(i) for e,i in zip(engines, lowest_counts)]

# # XXX Make the normal_trunc look like normal with 2 clusters, and make the arg
# # that we have better model by reassigning weight to the interval.
# # states_counts[-1].transition(kernels=['rows'],N=111)

# # Compute predictive densities.
# grid = 100
# rowids = [(-1,)] * grid
# evidences = [[] for _ in xrange(len(rowids))]

# # Compute normal predictive.
# e = engines[names.index('normal')]
# queries = [[(1,i)] for i in np.linspace(0.01, 359.8, grid)]
L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
logpdfs_normal = [logmeanexp(L[:,i]) for i in xrange(grid)]

# # Compute normaltrunc predictive.
# e = engines[names.index('normal_trunc')]
# queries = [[(1,i)] for i in np.linspace(0.01, 359.8, grid)]
# L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
# logpdfs_trunc = [logmeanexp(L[:,i]) for i in xrange(grid)]

# # Compute vonmises predictive.
# e = engines[names.index('vonmises')]
# queries = [[(1,i)] for i in np.linspace(0.01, 2*np.pi-0.01, grid)]
# L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
# logpdfs_vonmises = [logmeanexp(L[:,i]) for i in xrange(grid)]

# # Compute beta_uc predictive.
# e = engines[names.index('beta_uc')]
# queries = [[(1,i)] for i in np.linspace(0.01, 0.99, grid)]
# L = np.asarray(e.logpdf_bulk(rowids, queries, evidences=evidences))
# logpdfs_beta = [logmeanexp(L[:,i]-np.log(scale)) for i in xrange(grid)]
