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

import multiprocessing as mp
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, f1_score

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

# Load the engine and individual states.
print 'Loading engine ...'
satellites = Engine.from_pickle(file(
    'resources/20160219-1737-satellites.engine','r'))

print 'Loading states ...'
state = satellites.get_state(0)

def impute_missing_cell(state, rowid, col):
    x = state.X[rowid,col]
    if np.isnan(x):
        samples = state.simulate(rowid, [col], N=100)
        if state.dims(col).is_numeric():
            x = np.median(samples)
        else:
            samples = samples.reshape(1,-1)[0]
            x = Counter(samples).most_common()[0][0]
    return x

def impute_missing_row(state, rowid):
    return [impute_missing_cell(state, rowid, col) for col in
        xrange(state.n_cols())]

def impute_missing_dataset(state):
    V = [impute_missing_row(state, rowid) for rowid in xrange(state.n_rows())]
    return np.asarray(V)

def test_train_split_rows(D, seed=0, percent=.1):
    rng = np.random.RandomState(seed)
    nans = np.sum(np.isnan(D), axis=1)
    nan_rows = np.nonzero(nans)[0]
    valid_rows = np.nonzero(np.logical_not(nans))[0]
    # Split.
    test_rows = rng.choice(
        valid_rows, size=int(percent*len(valid_rows)), replace=0)
    train_rows = np.setdiff1d(valid_rows, test_rows)
    return train_rows, test_rows, nan_rows

def build_discretes(regressors):
    return {regressors.index(r):len(valmap[r]) for r in regressors
        if r in valmap}

def dummy_code(X, regressors):
    discretes = build_discretes(regressors)
    return np.asarray([du.dummy_code(x, discretes) for x in X])

# Experiments interested in running.
# Purpose, Anticipated_lifetime, Power_watts
# classification, regression, regression.
# Type of orbit with Random Forest and no latent mixture.

def random_forest_experiment_vanilla(seed=0, percent=.6):
    target = 'Type_of_Orbit'
    regressors = ['Eccentricity', 'Period_minutes', 'Launch_Mass_kg',
    'Dry_Mass_kg','Power_watts', 'Date_of_Launch', 'Anticipated_Lifetime']

    idx = [columns.index(s) for s in [target]+regressors]
    discretes = build_discretes(regressors)

    # Split original dataset.
    train, test, nan = test_train_split_rows(
        T[:,idx], seed=seed, percent=percent)

    # Imputation using the first state.
    Y_train = T[train,idx[0]]
    X_train = T[train][:,idx[1:]]

    Y_test = T[test,idx[0]]
    X_test = T[test][:,idx[1:]]

    rf = RandomForestClassifier(random_state=np.random.RandomState(0))
    rf.fit(X_train,Y_train)

    Y_pred = rf.predict(X_test)
    return Y_test, Y_pred

def random_forest_experiment_mixture(seed=0, percent=.6, cc=False):
    target = 'Type_of_Orbit'
    regressors = ['Eccentricity', 'Period_minutes', 'Launch_Mass_kg',
    'Dry_Mass_kg', 'Power_watts', 'Date_of_Launch', 'Anticipated_Lifetime']

    idx = [columns.index(s) for s in [target]+regressors]
    discretes = build_discretes(regressors)

    # Split original dataset.
    train, test, nan = test_train_split_rows(
        T[:,idx], seed=seed, percent=percent)

    # Obtain the partition.
    Zr = state.views[0].Zr

    def row_partition_data(rowids, X, Y, Zr):
        nk = len(set(Zr))
        assert len(rowids) == len(X) == len(Y)
        X_p = [[] for _ in xrange(nk)]
        Y_p = [[] for _ in xrange(nk)]
        for i, r in enumerate(rowids):
            X_p[Zr[r]].append(X[i])
            Y_p[Zr[r]].append(Y[i])
        return X_p, Y_p

    Y_train = T[train,idx[0]]
    X_train = T[train][:,idx[1:]]

    X_p, Y_p = row_partition_data(train, X_train, Y_train, Zr)

    RF = {}

    for i in xrange(len(X_p)):
        if len(X_p[i]) > 0:
            RF[i] = RandomForestClassifier(
                random_state=np.random.RandomState(0)).fit(X_p[i], Y_p[i])

    Y_test = T[test,idx[0]]
    X_test = T[test][:,idx[1:]]

    Y_pred = np.zeros(len(test))
    for (i, r) in enumerate(test):
        if Zr[r] not in RF or cc:
            samples = state.simulate(r, [columns.index(target)], N=50)
            y = Counter(samples.reshape(1,-1)[0]).most_common()[0][0]
        else:
            y = RF[Zr[r]].predict([X_test[i]])[0]
        Y_pred[i] = y

    return Y_test, Y_pred

def random_forest_experiment_cc(seed=0, percent=.6):
    return random_forest_experiment_mixture(seed=seed, percent=percent, cc=1)

def dispatch_experiment(rf_func):
    percents = np.linspace(.5, .95)
    seeds = range(10)

    def exp_ps(s,p):
        Yt, Yp = rf_func(seed=s, percent=p)
        return Yt, Yp

    def exp_s(s):
        return [exp_ps(s,p) for p in percents]

    # results[i,j,0] are the Ytest for seed i, percent j.
    # results[i,j,1] are the Ypred for seed i, percent j.
    return map(exp_s, seeds)

def compute_accuracy_seed(Yts, Yps):
    return [np.sum(Yt==Yp)/float(len(Yt)) for (Yt, Yp) in zip(Yts,Yps)]

# Takes as input the results matrix from dispatch experiment and an evaluator.
def evaluate_all(R, evaluator):
    return [evaluator(Yts, Yps) for (Yts, Yps) in zip(R[:,:,0], R[:,:,1])]

n_train = [81, 80, 78, 77, 75, 74, 72, 71, 69, 68, 66, 65, 63, 62, 60, 59, 57,
56, 54, 53, 51, 50, 48, 47, 46, 44, 43, 41, 40, 38, 37, 35, 34, 32, 31, 29, 28,
26, 25, 23, 22, 20, 19, 17, 16, 14, 13, 12, 10, 9,]

# Plots the output of evaluate_all for mix, van, cc.
def plot_comparison(mix, van, cc, yaxis):
    colors = {'CC':'b', 'RF':'g', 'CC mixture of RF':'r'}
    fig, ax = plt.subplots()
    for exp, lab in zip([mix, van, cc], ['CC mixture of RF', 'RF', 'CC']):
        avg = np.mean(exp, axis=0)[::-1]
        top = np.percentile(exp, 25, axis=0)[::-1]
        low = np.percentile(exp, 75, axis=0)[::-1]
        xs = n_train[::-1]
        c = colors[lab]
        ax.errorbar(
            xs, avg, yerr=[avg-low, top-avg], fmt='--o', color=c, label=lab)

    ax.set_title('SIMULATE Type_of_Orbit GIVEN * ON TEST SET', fontweight='bold')
    ax.set_xlabel('Number of observations', fontweight='bold')
    ax.set_ylabel(yaxis, fontweight='bold')
    ax.grid()
    ax.legend(loc='upper left', framealpha=0)


# CREATE THE SETS
# print 'Dispatching mixture ...'
# RM = dispatch_experiment(random_forest_experiment_mixture)
# print 'Dispatching vanilla ...'
# RV = dispatch_experiment(random_forest_experiment_vanilla)
# print 'Dispatching crosscat ...'
# RC = dispatch_experiment(random_forest_experiment_cc)

# np.save('resources/RM', RM)
# np.save('resources/RV', RV)
# np.save('resources/RC', RC)

# LOAD FROM CACHE
RM = np.load('resources/RM.npy')
RV = np.load('resources/RV.npy')
RC = np.load('resources/RC.npy')

acc_m = np.asarray(evaluate_all(RM, compute_accuracy_seed))
acc_c = np.asarray(evaluate_all(RC, compute_accuracy_seed))
acc_v = np.asarray(evaluate_all(RV, compute_accuracy_seed))


plot_comparison(acc_m, acc_v, acc_c, 'Prediction Accuracy')
