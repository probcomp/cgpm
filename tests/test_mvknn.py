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
import os

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scipy.stats import ks_2samp
from sklearn.neighbors import KDTree

from cgpm.knn.mvknn import MultivariateKnn
from cgpm.utils import general as gu
from cgpm.utils import test as tu


O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'nominal'


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

    # Neighbor search need constraints.
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], None)
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], {})

    # Bad category 199.
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], {2:199, 5:.8})

    # Check returned K and dimension are correct varying targets/constraints.
    for targets, constraints in [
            ([0,1], {5:.8}),
            ([0,1,7], {5:.8}),
            ([4], {5:.8, 7:0})
    ]:
        d, nh = knn._find_neighborhoods(targets, constraints)
        assert len(nh) == K
        for n in nh:
            assert 1 <= len(n) <= K
            assert d[n].shape[1] == len(targets)

    # Dimension 10 has only 4 non-nan values, so K=5 will fail.
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], {10:.8})
    with pytest.raises(ValueError):
        knn._find_neighborhoods([10,1], {5:.8})

    # Now furnish dimension 10 with one additional non-nan value and ensure the
    # K=5 neighbors are the only possible ones i.e 0, 96, 97, 98, 99.
    knn.data[0][10] = 1

    def test_found_expected(dataset, neighborhoods, expect):
        for e in expect:
            match = False
            for n in neighborhoods:
                match = match or any(np.allclose(dataset[i], e) for i in n)
            assert match

    d_found, n_found = knn._find_neighborhoods([0,1], {10:.8})
    expected = [np.asarray(knn.data[r])[[0,1]]
        for r in [0, 96, 97, 98, 99]]
    test_found_expected(d_found, n_found, expected)

    d_found, n_found = knn._find_neighborhoods([10,1], {5:.8})
    expected = [np.asarray(knn.data[r])[[10,1]]
        for r in [0, 96, 97, 98, 99]]
    test_found_expected(d_found, n_found, expected)

    # Now make sure an exact match is in the nearest neighbor.
    z = knn.data[19]

    # # First crash since z contains a nan.
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], dict(zip(outputs[2:], z[2:])))

    # Now make sure that z is its own nearest neighbor.
    z_targets = [0,1]
    z_constraints = {o:v for o,v in zip(outputs[2:], z[2:]) if not np.isnan(v)}
    d_found, n_found = knn._find_neighborhoods(z_targets, z_constraints)
    test_found_expected(d_found, n_found, [z[:2]])


def test_perigee_period_given_apogee():
    # This test uses KNN to answer two BQL queries.
    # SIMULATE perigee_km, period_minutes GIVEN apogee_km = 500;
    # SIMULATE apogee_km, period_minutes;
    # The outputs of the query are scattered on a plot.

    rng = gu.gen_rng(1)

    # Load the satellites dataset.
    filename = os.path.join(
        os.path.dirname(__file__), 'graphical/resources/satellites.csv')
    satellites = pd.read_csv(filename)

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
    _fig, ax = plt.subplots()

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
    _fig, ax = plt.subplots()

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


# XXX The following three tests are very similar to test_normal_categorical. The
# two tests can be merged easily and it should be done to reduce duplication.

def generate_real_nominal_data(N, rng=None):
    # Generates a bivariate dataset, where the first variable x is real-valued
    # and the second variable z is nominal with 6 levels. The real variable's
    # mean is determined by the value of z, where there are three means
    # corresponding to levels [(0,1), (2,3), (4,5)].

    if rng is None: rng = gu.gen_rng(0)
    T, Zv, Zc = tu.gen_data_table(
        N, [1], [[.3, .5, .2]], ['normal'], [None], [.95], rng=rng)
    data = np.zeros((N, 2))
    data[:,0] = T[0]
    indicators = [0, 1, 2, 3, 4, 5]
    counts = {0:0, 1:0, 2:0}
    for i in xrange(N):
        k = Zc[0][i]
        data[i,1] = 2*indicators[k] + counts[k] % 2
        counts[k] += 1
    return data, indicators


@pytest.fixture(scope='module')
def knn_xz():
    # Learns an MvKnn on the dataset generated by generate_real_nominal_data
    # and returns the fixture for use in the next three tests.

    N_SAMPLES = 250
    data, indicators = generate_real_nominal_data(N_SAMPLES)
    K = MultivariateKnn(
        [0,1], None,
        K=20,
        M=5,
        distargs={'outputs': {ST: [N, C], SA:[{}, {'k': len(indicators)}]}},
        rng=gu.gen_rng(0))
    for rowid, x in enumerate(data):
        K.incorporate(rowid, {0:x[0], 1:x[1]})
    K.transition()
    return K


def test_joint(knn_xz):
    # Simulate from the joint distribution of x,z (see
    # generate_real_nominal_data) and perform a KS tests at each of the
    # subpopulations at the six levels of z.

    data = np.asarray(knn_xz.data.values())
    indicators = sorted(set(data[:,1].astype(int)))
    joint_samples = knn_xz.simulate(-1, [0,1], N=len(data))
    _, ax = plt.subplots()
    ax.set_title('Joint Simulation')
    for t in indicators:
        # Plot original data.
        data_subpop = data[data[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data for indicator t.
        samples_subpop = [j[0] for j in joint_samples if j[1] == t]
        ax.scatter(
            np.add([t]*len(samples_subpop), .25), samples_subpop,
            color=gu.colors[t])
        # KS test.
        pvalue = ks_2samp(data_subpop[:,0], samples_subpop)[1]
        assert .05 < pvalue
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_indicator(knn_xz):
    # Simulate from the conditional distribution of x|z (see
    # generate_real_nominal_data) and perfrom a KS tests at each of the
    # subpopulations at the six levels of z.

    data = np.asarray(knn_xz.data.values())
    indicators = sorted(set(data[:,1].astype(int)))
    _, ax = plt.subplots()
    ax.set_title('Conditional Simulation Of X Given Indicator Z')
    for t in indicators:
        # Plot original data.
        data_subpop = data[data[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data.
        samples_subpop = [s[0] for s in
            knn_xz.simulate(-1, [0], constraints={1:t}, N=len(data_subpop))]
        ax.scatter(
            np.repeat(t, len(data_subpop)) + .25,
            samples_subpop, color=gu.colors[t])
        # KS test.
        pvalue = ks_2samp(data_subpop[:,0], samples_subpop)[1]
        assert .1 < pvalue
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_real(knn_xz):
    # Simulate from the conditional distribution of z|x (see
    # generate_real_nominal_data) and plot the frequencies of the simulated
    # values.

    data = np.asarray(knn_xz.data.values())
    indicators = sorted(set(data[:,1].astype(int)))
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Conditional Simulation Of Indicator Z Given X', size=20)
    # Compute representative data sample for each indicator.
    means = [np.mean(data[data[:,1]==t], axis=0)[0] for t in indicators]
    for mean, indicator, ax in zip(means, indicators, axes.ravel('F')):
        samples_subpop = [s[1] for s in
            knn_xz.simulate(-1, [1], constraints={0:mean}, N=len(data))]
        # Plot a histogram of the simulated indicator.
        ax.hist(samples_subpop, color='g', alpha=.4)
        ax.set_title('True Indicator Z %d' % indicator)
        ax.set_xlabel('Simulated Indicator Z')
        ax.set_xticks(indicators)
        ax.set_ylabel('Frequency')
        ax.set_ylim([0, ax.get_ylim()[1]+10])
        ax.grid()
        # Check that the simulated indicator agrees with true indicator.
        true_ind_a = indicator
        true_ind_b = indicator-1  if indicator % 2 else indicator+1
        counts = np.bincount(samples_subpop)
        frac = sum(counts[[true_ind_a, true_ind_b]])/float(sum(counts))
        assert .8 < frac
