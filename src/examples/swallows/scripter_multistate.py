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
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import logsumexp

from gpmcc.engine import Engine

def logmeanexp(array):
    from scipy.misc import logsumexp
    return logsumexp(array) - np.log(len(array))

def compute_mutual_information(a, b):
    assert len(a) == len(b)
    N = float(len(a))
    # Joint distribution.
    pab = np.zeros((2,2))
    pab[1,1] = np.sum(np.logical_and(a==1, b==1)) / N
    pab[0,0] = np.sum(np.logical_and(a==0, b==0)) / N
    pab[0,1] = np.sum(np.logical_and(a==0, b==1)) / N
    pab[1,0] = np.sum(np.logical_and(a==1, b==0)) / N
    # Marginal distribution.
    pa = [np.sum(a==0)/N, np.sum(a==1)/N]
    pb = [np.sum(b==0)/N, np.sum(b==1)/N]

    mi = 0
    for i,j in itertools.product([0,1], [0,1]):
        if pab[i,j] == 0: continue
        mi += pab[i,j] * (np.log2(pab[i,j]) - np.log2(pa[i]*pb[j]))
    return mi


# ------- LOADING -------- #

# Load the overall dataset.
df = pd.read_csv('swallows.csv')
df.replace('control', 0, inplace=1)
df.replace('shifted', 1, inplace=1)
D = np.asarray(df)

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

# Transition the beta to plot.
s_beta = engines[2][0].get_state(20)
s_beta.transition(kernels=['column_params'])
s_beta.transition(kernels=['column_params','column_hypers'], N=100)
s_beta.transition(kernels=['column_params','column_hypers'], N=100)
s_beta.transition(kernels=['column_params'], N=10)
s_beta.transition(kernels=['column_params'], N=5)
s_beta.transition(kernels=['column_params'], N=5)
s_beta.transition(kernels=['column_params'], N=5)

# Number of states per engine.
num_states = engines[0][0].num_states


# ------- COMPUTING -------- #

# Compute the marginals.
if False:
    marginals = np.asarray(
        [[e.logpdf_marginal() for e in engines[i]] for i in xrange(num_dists)])

    # XXX Adjust beta for scaling by 2pi.
    marginals[names.index('beta_uc')] = \
        marginals[names.index('beta_uc')] - len(train_sets[0])*np.log(2*np.pi)

    # Find states with highest marginals.
    highest_marginals = np.argmax(marginals, axis=2)

    # Pool all the marginals into array.
    all_marginals = np.vstack([marginals[:,i,:].T for i in xrange(num_dists)])
    np.save('all_marginals', all_marginals)
else:
    all_marginals = np.load('all_marginals.npy')

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

# Compute the predictive likelihood.
all_logpdfs = np.asarray(all_logpdfs)
predictive = logsumexp(all_logpdfs[:,:,:,:], axis=2)-np.log(num_states)
predictive = np.swapaxes(predictive,0,1).reshape((4,60)).T

# Find the true labels of treatment.
all_samples = np.asarray(all_samples)
for T in test_sets:
    true_samples.append([])
    for row in T.iterrows():
        treatment, heading = row[1][0], row[1][1]
        treatment = 0 if treatment == 'control' else 1
        true_samples[-1].append(treatment)

true_samples = np.asarray(true_samples).T

# Compute mutual information with labels and predictions as a function of
# classification threshold.
predictions = np.swapaxes(all_samples,0,1).reshape((4,60)).T
thresholds = np.linspace(0,1,1000)
curves = np.zeros((num_dists, len(thresholds)))
for i, t in enumerate(thresholds):
    pred_t = predictions < t
    for d in xrange(num_dists):
        curves[d][i] = compute_mutual_information(
            true_samples.ravel(), pred_t[:,d])

# ------- PLOTTING -------- #

# Histogram the data.
width = (2*np.pi) / 100

fig = plt.figure()
fig.add_subplot(121, projection='polar')

fig.axes[0].set_yticklabels([])
fig.axes[0].hist(
    D[D[:,0]==0][:,1]*np.pi/180., width=width, bottom=1, alpha=.4,
    color='r', bins=100, normed=1, label='Local Field')
fig.axes[0].legend(framealpha=0).draggable()

fig.add_subplot(122, projection='polar')
fig.axes[1].set_yticklabels([])
fig.axes[1].hist(
    D[D[:,0]==1][:,1]*np.pi/180., width=width, bottom=1, alpha=.4,
    color='g',bins=100, normed=1, label='Shifted Field')
fig.axes[1].legend(framealpha=0).draggable()

# Plot samples from the posterior over distributions.
sample_engines = {
    'normal': [engines[0][0].get_state(2), engines[0][1].get_state(13)],
    'trunc': [engines[1][0].get_state(7), engines[1][1].get_state(3)],
    'vonmises': [engines[3][0].get_state(20), engines[3][1].get_state(26)],
    'beta' : [s_beta, engines[2][1].get_state(8)]
}
for i in xrange(2):
    fig, axes = plt.subplots(nrows=2, ncols=2)

    # s_norm = engines[0][0].get_state(2)
    s_normal = sample_engines['normal'][i]
    ax_normal = s_normal.dims[1].plot_dist(s_normal.X[:,1],
        Y=np.linspace(-1,7,100), ax=axes[0][0])
    ax_normal.set_xlim([-1, 7])
    ax_normal.set_xticklabels([])
    ax_normal.set_ylim([0, 0.35])
    ax_normal.set_ylabel('Density')
    ax_normal.grid()

    # s_trunc = engines[1][0].get_state(7)
    s_trunc = sample_engines['trunc'][i]
    ax_trunc = s_trunc.dims[1].plot_dist(s_trunc.X[:,1], ax=axes[0][1])
    ax_trunc.set_xlim([-1, 7])
    ax_trunc.set_xticklabels([])
    ax_trunc.set_ylim([0, 0.35])
    ax_trunc.set_yticklabels([])
    ax_trunc.grid()

    # s_vonmises = engines[3][0].get_state(20)
    s_vonmises = sample_engines['vonmises'][i]
    ax_vonmises = s_vonmises.dims[1].plot_dist(s_vonmises.X[:,1], ax=axes[1][0])
    ax_vonmises.set_xlim([-1, 7])
    ax_vonmises.set_ylim([0, 0.35])
    ax_vonmises.grid()

    s_beta = sample_engines['beta'][i]
    ax_beta = s_beta.dims[1].plot_dist(s_beta.X[:,1], ax=axes[1][1])
    ax_beta.set_xlim([-1/(2*np.pi), 7/(2*np.pi)])
    ax_beta.set_xticks([-1./6,0,1./6,2./6,3./6,4./6,5./6,6./6,7./6])
    ax_beta.set_xticklabels(range(-1,8))
    ax_beta.set_ylim(0, 2*np.pi*0.35)
    ax_beta.set_yticklabels([])
    ax_beta.grid()

# yticklabels = ['{:1.2f}'.format(yt) for yt in ax_beta.get_yticks()/(2*np.pi)]
# ax_beta.set_yticklabels(yticklabels)

# NORMAL 0 1 13
# TRUNC  1 1 3
# VONMISES 3 1 26
# BETA 2 1 8

# Boxplot the predictives.
fig, ax = plt.subplots(nrows=2, ncols=1)

ax[0].boxplot(all_marginals, labels=names)
ax[0].set_ylabel('Marginal Likelihood (Estimates)')
ax[0].set_ylim([-1000, 0])
ax[0].grid()

ax[1].boxplot(predictive, labels=names)
ax[1].set_ylabel('Predictive Likelihood (Estimates)')
ax[1].grid()

# Plot mutual information as a function of threshold.
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0][0].plot(thresholds, curves[0])
axes[0][0].set_title('normal')
axes[0][0].set_xticklabels([])
axes[0][0].set_ylabel('MI (Bits)')

axes[0][1].plot(thresholds, curves[1], color='g')
axes[0][1].set_title('normal_trunc')
axes[0][1].set_yticklabels([])
axes[0][1].set_xticklabels([])

axes[1][0].plot(thresholds, curves[2], color='r')
axes[1][0].set_title('vonmises')
axes[1][0].set_xlabel('Classification Threshold')
axes[1][0].set_ylabel('MI (Bits)')

axes[1][1].plot(thresholds, curves[3], color='b')
axes[1][1].set_title('beta')
axes[1][1].set_yticklabels([])
axes[1][1].set_xlabel('Classification Threshold')

for a in axes.ravel():
    a.set_ylim([0, 0.08])
    a.grid()
    a.set_yticks([0, .02, .04, .06, .08])
