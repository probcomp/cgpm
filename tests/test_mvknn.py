# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scipy.stats import chisquare
from scipy.stats import ks_2samp
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor

from cgpm.knn.mvknn import MultivariateKnn
from cgpm.utils import general as gu
from cgpm.utils import test as tu


O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'categorical'


def test_initialize():
    # This test ensures that MvKnn raises on bad initialize arguments.
    # Typical initialization.
    MultivariateKnn(
        outputs=[0, 1], inputs=None, K=2,
        distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # No inputs allowed.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=[2], K=2,
            distargs={O: {ST:[N, C], SA: [{}, {'k': 2}]}})
    # At least two output.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[], inputs=[], K=2,
            distargs={O: {ST: [], SA:[]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[1], inputs=[], K=2,
            distargs={O: {ST: [], SA:[]}})
    # Unique outputs.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 0], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure outputs in distargs.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs=None)
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={'output': {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure stattypes and statargs in distargs['outputs]'
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {'stattype': [N, C], SA :[{}, {'k': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], 'eland': [{}, {'k': 2}]}})
    # Ensure stattypes correct length.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C, N], SA: [{}, {'k': 2}]}})
    # Ensure statargs correct length.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, None, {'k': 2}]}})
    # Ensure number of categories provided as k.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, {'h': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, {}]}})
    # Missing number of nearest neighbors K.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Bad number of nearest neighbors K.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=0,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=-1,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})


def test_find_neighborhoods():
    # This test target _find_neighborhoods from MultivariateKnn. See the
    # inline comments for the description of each test.

    # Generate a high dimensional dataset with mixed numerical/categorical.
    rng = gu.gen_rng(1)

    outputs = range(11)
    inputs = None
    K = 5
    stattypes = [N, N, C, N, N, N, C, N, N, C, N]
    statargs = [{}, {}, {'k':7}, {}, {}, {}, {'k':1}, {}, {}, {'k':5}, {}]
    distargs = {'outputs': {'stattypes': stattypes, 'statargs': statargs}}

    X = rng.rand(100, 11)
    X[:,2] = rng.choice(range(statargs[2]['k']), size=100)
    X[:,6] = rng.choice(range(statargs[6]['k']), size=100)
    X[:,9] = rng.choice(range(statargs[9]['k']), size=100)
    X[:96,10] = np.nan

    knn = MultivariateKnn(outputs, inputs, K=K, distargs=distargs)

    for i, x in enumerate(X):
        knn.incorporate(i, dict(zip(outputs, x)))

    assert knn.N == len(X)

    # Neighbor search need evidence.
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence=None)
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence={})

    # Bad category 199.
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence={2:199, 5:.8})

    # Check the returned K and dimension are correct varying query/evidence.
    for q, e in [
            ([0,1], {5:.8}),
            ([0,1,7], {5:.8}),
            ([4], {5:.8, 7:0})
    ]:
        d, nh = knn._find_neighborhoods(query=q, evidence=e)
        assert len(nh) == K
        for n in nh:
            assert 1 <= len(n) <= K
            assert d[n].shape[1] == len(q)

    # Dimension 10 has only 4 non-nan values, so K=5 will fail.
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence={10:.8})
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[10,1], evidence={5:.8})

    # Now furnish dimension 10 with one additional non-nan value and ensure the
    # K=5 neighbors are the only possible ones i.e 0, 96, 97, 98, 99.
    knn.data[0][10] = 1

    def test_found_expected(dataset, neighborhoods, expect):
        for e in expect:
            match = False
            for n in neighborhoods:
                match = match or any(np.allclose(dataset[i], e) for i in n)
            assert match

    d_found, n_found = knn._find_neighborhoods(query=[0,1], evidence={10:.8})
    expected = [np.asarray(knn.data[r])[[0,1]]
        for r in [0, 96, 97, 98, 99]]
    test_found_expected(d_found, n_found, expected)

    d_found, n_found = knn._find_neighborhoods(query=[10,1], evidence={5:.8})
    expected = [np.asarray(knn.data[r])[[10,1]]
        for r in [0, 96, 97, 98, 99]]
    test_found_expected(d_found, n_found, expected)

    # Now make sure an exact match is in the nearest neighbor.
    z = knn.data[19]

    # # First crash since z contains a nan.
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], dict(zip(outputs[2:], z[2:])))

    # Now make sure that z is its own nearest neighbor.
    z_query = [0,1]
    z_evidence = {o:v for o,v in zip(outputs[2:], z[2:]) if not np.isnan(v)}
    d_found, n_found = knn._find_neighborhoods(z_query, z_evidence)
    test_found_expected(d_found, n_found, [z[:2]])


def test_perigee_period_given_apogee():
    rng = gu.gen_rng(1)

    # Load the satellites dataset.
    satellites = pd.read_csv('resources/satellites.csv')

    # Extract target columns of interest.
    D = satellites[['Apogee_km', 'Perigee_km', 'Period_minutes']].dropna()
    X = np.asarray(D)

    # Extract the nearest neighbors given A=500.
    tree = KDTree(X[:,0].reshape(-1,1))
    _, neighbors = tree.query([[500]], k=20)
    perigees = X[neighbors[0][:10],1]
    periods = X[neighbors[0][:10],2]

    # Learn the joint distribution by assuming P,T|A are independent.
    perigees_ind = rng.normal(np.mean(perigees), np.std(perigees), size=20)
    periods_ind = rng.normal(np.mean(periods), np.std(perigees), size=20)

    # Create a KNN.
    distargs = {
    'outputs': {
        'stattypes': ['numerical', 'numerical', 'numerical'],
        'statargs': [{}, {}, {}]
    }}
    knn = MultivariateKnn([0,1,2], None, distargs=distargs, K=30, rng=rng)
    for i, row in enumerate(X):
        knn.incorporate(i, dict(zip([0,1,2], row)))

    # Sample from the dependent KNN.
    samples_dep = knn.simulate(-1, [1,2], {0: 500}, N=20)
    logpdfs = [knn.logpdf(-1, s, {0: 500}) for s in samples_dep]
    assert all(not np.isinf(l) for l in logpdfs)

    # Create an axis.
    fig, ax = plt.subplots()

    # Scatter the actual neighborhood.
    ax.scatter(perigees, periods, color='b', label='Actual Satellites')

    # Plot the independent knn.
    ax.scatter(
        perigees_ind, periods_ind, color='r', alpha=.5,
        label='Independent KNN')

    # Plot the dependent knn.
    ax.scatter(
        [s[1] for s in samples_dep], [s[2] for s in samples_dep],
        color='g', alpha=.5, label='Dependent KNN')

    # Prepare the axes.
    ax.set_title(
        'SIMULATE Perigee_km, Period_minutes GIVEN Apogee_km = 500',
        fontweight='bold')
    ax.set_xlabel('Perigee', fontweight='bold')
    ax.set_ylabel('Period', fontweight='bold')
    ax.grid()
    ax.legend(framealpha=0, loc='upper left')

    # Now simulate from the joint distributions of apogee, perigee.
    samples_joint = knn.simulate(-1, [0,2], N=100)

    # Create an axis.
    fig, ax = plt.subplots()

    # Scatter the actual data.
    ax.scatter(X[:,0], X[:,2], color='b', label='Actual Satellites')

    # Scatter the simulated data.
    ax.scatter(
        [s[0] for s in samples_joint], [s[2] for s in samples_joint],
        color='r', label='Dependent KNN')

    # Prepare the axes.
    ax.set_title(
        'SIMULATE period_minutes, apogee_km LIMIT 500', fontweight='bold')
    ax.set_xlabel('Apogee', fontweight='bold')
    ax.set_ylabel('Period', fontweight='bold')
    ax.set_xlim([-500, 50000])
    ax.set_ylim([-100, 1800])
    ax.grid()
    ax.legend(framealpha=0, loc='upper left')

    # Reveal!
    plt.close('all')


def test_serialize():
    rng = gu.gen_rng(1)

    data = rng.rand(20, 5)
    data[:10,-1] = 0
    data[10:,-1] = 1

    knn = MultivariateKnn(
        range(5),
        None,
        K=10,
        distargs={
            'outputs': {
                'stattypes': [
                    'numerical',
                    'numerical',
                    'numerical',
                    'numerical',
                    'categorical'
                ],
                'statargs': [
                    {},
                    {},
                    {},
                    {},
                    {'k':1}
        ]}},
        rng=rng)

    for rowid, x in enumerate(data):
        knn.incorporate(rowid, dict(zip(range(5), x)))

    knn.transition()

    metadata_s = json.dumps(knn.to_metadata())
    metadata_l = json.loads(metadata_s)

    modname = importlib.import_module(metadata_l['factory'][0])
    builder = getattr(modname, metadata_l['factory'][1])
    knn2 = builder.from_metadata(metadata_l, rng=rng)

    # Variable indexes.
    assert knn2.outputs == knn.outputs
    assert knn2.inputs == knn.inputs
    # Distargs.
    assert knn2.get_distargs() == knn.get_distargs()
    assert knn2.get_distargs() == knn.get_distargs()
    # Dataset.
    assert knn2.data == knn.data
    assert knn2.N == knn.N
    # Bandwidth params.
    assert knn2.stattypes == knn.stattypes
    assert knn2.statargs == knn.statargs
