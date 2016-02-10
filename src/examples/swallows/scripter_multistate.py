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

def logmeanexp(array):
    from scipy.misc import logsumexp
    return logsumexp(array) - np.log(len(array))

# Load the overall dataset.
df = pd.read_csv('swallows.csv')
df.replace('control', 0, inplace=1)
df.replace('shifted', 1, inplace=1)
T = np.asarray(df)

# Histogram the data.
width = (2*np.pi) / 100

fig = plt.figure()
fig.add_subplot(111, projection='polar')

fig.axes[0].set_yticklabels([])
fig.axes[0].hist(
    T[T[:,0]==0][:,1]*np.pi/180., width=width, bottom=1, alpha=.4,
    color='r', bins=100, normed=1, label='Local Field')

fig.axes[0].set_yticklabels([])
fig.axes[0].hist(
    T[T[:,0]==1][:,1]*np.pi/180., width=width, bottom=1, alpha=.4,
    color='g',bins=100, normed=1, label='Shifted Field')

fig.axes[0].legend(framealpha=0).draggable()

# Load the train and test sets.
print 'Loading test/train splits ...'
test_sets = []
train_sets = []

for f in os.listdir('splits'):
    (num, kind, _) = f.split('-')
    if kind == 'train':
        train_sets.append(pd.read_csv('splits/%s' % f, index_col=0))
    else:
        test_sets.append(pd.read_csv('splits/%s' % f, index_col=0))

# Useful parameters.
names = ['normal', 'normal_trunc', 'beta_uc', 'vonmises']
num_dists = len(names)
num_sets = len(test_sets)
num_test_samples = len(test_sets[0])

# Load the engines.
filenames = [
    '20160209-223338-normal-0-swallows.engine',
    '20160209-223804-normal-1-swallows.engine',
    '20160209-224224-normal-2-swallows.engine',
    '20160209-224705-normal-3-swallows.engine',
    '20160209-225108-normal-4-swallows.engine',
    '20160209-225353-normal_trunc-0-swallows.engine',
    '20160209-233249-normal_trunc-2-swallows.engine',
    '20160209-235254-normal_trunc-3-swallows.engine',
    '20160210-001307-normal_trunc-4-swallows.engine',
    '20160209-231324-normal_trunc-1-swallows.engine',
    '20160209-224209-beta_uc-0-swallows.engine',
    '20160209-225253-beta_uc-1-swallows.engine',
    '20160209-230333-beta_uc-2-swallows.engine',
    '20160209-231404-beta_uc-3-swallows.engine',
    '20160209-232452-beta_uc-4-swallows.engine',
    '20160209-223228-vonmises-0-swallows.engine',
    '20160209-223546-vonmises-1-swallows.engine',
    '20160209-223909-vonmises-2-swallows.engine',
    '20160209-224159-vonmises-3-swallows.engine',
    '20160209-224539-vonmises-4-swallows.engine',
    ]

# Store engines in memory.
print 'Loading engines ...'
engines = [ [None for _ in xrange(num_sets)] for _ in xrange(num_dists)]
for f in filenames:
    with open('resources/%s' % f,'r') as infile:
        engine = Engine.from_pickle(infile)
        (_, _, dist, num, _) = f.split('-')
        engines[names.index(dist)][int(num)] = engine

# Number of states per engine.
num_states = engines[0][0].num_states

# Compute the marginals.
if False:
    marginals = np.asarray(
        [[e.logpdf_marginal() for e in engines[i]] for i in xrange(num_dists)])

    # XXX Adjust beta for scaling by 2pi.
    marginals[names.index('beta_uc')] = \
        marginals[names.index('beta_uc')] - len(train_sets[0])*np.log(2*np.pi)

    # Find states with highest marginals.
    lowest_marginals = np.argmax(marginals, axis=2)

    # Pool all the marginals into array.
    all_marginals = np.vstack([marginals[:,i,:].T for i in xrange(num_dists)])
    np.save('all_marginals', all_marginals)
else:
    all_marginals = np.load('all_marginals.npy')

# Boxplot.
_, ax = plt.subplots()
ax.boxplot(all_marginals, labels=names)
ax.set_ylabel('Marginal Likelihood (Estimates)')
ax.set_ylim([-1000, 0])
ax.grid()

# Compute the number of clusters in all the states.
cluster_counts = []
for dist in xrange(len(engines)):
    cluster_counts.append([])
    for train_set in xrange(num_sets):
        e = engines[dist][train_set]
        cluster_counts[-1].append(
            [len(set(e.metadata[i]['Zrcv'][e.metadata[i]['Zv'][1]]))
                for i in xrange(e.num_states)])

cluster_counts = np.asarray(cluster_counts)
lowest_counts = np.argmin(cluster_counts, axis=2)

# Lists for results.
all_logpdfs = []
all_samples = []
true_samples = []

# Compute predictive likelihood and posterior simulations.
if False:
    for i in xrange(num_sets):
        # Engines for this train/test split.
        test_engines = [engines[j][i] for j in xrange(len(names))]
        T = test_sets[i]
        rowids = [-1] * len(T)
        logpdf_queries = []
        simulate_queries = [(0,)] * len(T)
        simulate_evidences = []
        for row in T.iterrows():
            treatment, heading = row[1][0], row[1][1]
            treatment = 0 if treatment == 'control' else 1
            heading *= np.pi/180
            logpdf_queries.append([(0, treatment), (1, heading)])
            simulate_evidences.append([(1, heading)])

        print 'Computing predictives ...'
        logpdfs = []
        for i, te in enumerate(test_engines):
            print i
            Q = logpdf_queries
            if names[i] == 'beta_uc':
                Q = [[a,(b0,b1/(2*np.pi))] for [a,(b0,b1)] in logpdf_queries]
            r = te.logpdf_bulk(rowids, Q)
            if names[i] == 'beta_uc':
                r = r - np.log(2*np.pi)
            logpdfs.append(r)
        all_logpdfs.append(np.asarray(logpdfs))

        print 'Simulating treatment'
        samples = []
        for i, te in enumerate(test_engines):
            print i
            E = simulate_evidences
            if names[i] == 'beta_uc':
                E = [[(b0, b1/(2*np.pi))] for [(b0,b1)] in simulate_evidences]
            s = te.simulate_bulk(
                rowids, simulate_queries, evidences=E,
                Ns=[1]*(len(rowids)))
            s = np.asarray(s)
            s = np.sum(s.reshape(
                    num_states, num_test_samples), axis=0) / float(num_states)
            samples.append(s)
        all_samples.append(np.asarray(samples))
else:
    all_logpdfs = np.load('all_logpdfs.npy')
    all_samples = np.load('all_samples.npy')

# Convert engine output to np.
all_samples = np.asarray(all_samples)
all_logpdfs = np.asarray(all_logpdfs)

# Find the true labels of treatment.
for T in test_sets:
    true_samples.append([])
    for row in T.iterrows():
        treatment, heading = row[1][0], row[1][1]
        treatment = 0 if treatment == 'control' else 1
        true_samples[-1].append(treatment)

true_samples = np.asarray(true_samples).T
