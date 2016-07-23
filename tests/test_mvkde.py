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

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scipy.stats import ks_2samp

from cgpm.kde.mvkde import MultivariateKde
from cgpm.utils import general as gu
from cgpm.utils import test as gt


O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'categorical'


def test_initialize():
    # Typical initialization.
    MultivariateKde(
        outputs=[0, 1], inputs=None,
        distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # No inputs allowed.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=[2],
            distargs={O: {ST:[N, C], SA: [{}, {'k': 2}]}})
    # At least one output.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[], inputs=[],
            distargs={O: {ST: [], SA:[]}})
    # Unique outputs.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 0], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure outputs in distargs.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs=None)
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={'output': {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure stattypes and statargs in distargs['outputs]'
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {'stattype': [N, C], SA :[{}, {'k': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], 'eland': [{}, {'k': 2}]}})
    # Ensure stattypes correct length.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C, N], SA: [{}, {'k': 2}]}})
    # Ensure statargs correct length.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, None, {'k': 2}]}})
    # Ensure number of categories provided as k.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {'h': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {}]}})


def test_invalid_incorporate():
    # No evidence.
    kde = MultivariateKde(
        outputs=[0, 1], inputs=None,
        distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Missing query.
    with pytest.raises(ValueError):
        kde.incorporate(0, {})
    # Disallow evidence.
    with pytest.raises(ValueError):
        kde.incorporate(0, {0:1}, {1:2})
    # Unknown output var.
    with pytest.raises(ValueError):
        kde.incorporate(0, {0:1, 2:1})
    kde.incorporate(0, {0:1, 1:1})
    # Duplicate rowid.
    with pytest.raises(ValueError):
        kde.incorporate(0, {1:1})
    # Unspecified entry 0 should be nan
    kde.incorporate(1, {1:1})
    assert np.isnan(kde.data[1][0])


def uni_normal_1(N, rng):
    return rng.normal(-1, 1, size=N)
def uni_normal_2(N, rng):
    return rng.normal(-1, 4, size=N)
def uni_normal_3(N, rng):
    return rng.normal(-1, 16, size=N)
def uni_normal_4(N, rng):
    return rng.normal(10, 1, size=N)
def uni_normal_5(N, rng):
    return rng.normal(10, 4, size=N)
def uni_normal_6(N, rng):
    return rng.normal(10, 16, size=N)
def uni_normal_8(N, rng):
    return rng.normal(-13, 4, size=N)
def uni_normal_9(N, rng):
    return rng.normal(-13, 16, size=N)
def bi_normal_1(N, rng):
    counts = rng.multinomial(N, pvals=[.7,.3])
    return np.hstack((
        uni_normal_1(counts[0], rng),
        uni_normal_2(counts[1], rng)))
def bi_normal_2(N, rng):
    counts = rng.multinomial(N, pvals=[.6,.4])
    return np.hstack((
        uni_normal_5(counts[0], rng),
        uni_normal_8(counts[1], rng)))
def bi_normal_3(N, rng):
    counts = rng.multinomial(N, pvals=[.5,.5])
    return np.hstack((
        uni_normal_2(counts[0], rng),
        uni_normal_8(counts[1], rng)))
def bi_normal_4(N, rng):
    counts = rng.multinomial(N, pvals=[.5,.5])
    return np.hstack((
        uni_normal_6(counts[0], rng),
        uni_normal_1(counts[1], rng)))
def bi_normal_5(N, rng):
    counts = rng.multinomial(N, pvals=[.65,.45])
    return np.hstack((
        uni_normal_1(counts[0], rng),
        uni_normal_4(counts[1], rng)))

SAMPLES = [
    uni_normal_1,
    uni_normal_2,
    uni_normal_3,
    uni_normal_4,
    uni_normal_5,
    uni_normal_6,
    uni_normal_8,
    uni_normal_9,
    bi_normal_1,
    bi_normal_2,
    bi_normal_3,
    bi_normal_4,
    bi_normal_5,
]
N_SAMPLES = 100

@pytest.mark.parametrize('i', xrange(len(SAMPLES)))
def test_univariate_two_sample(i):
    rng = gu.gen_rng(2)
    # Synthetic samples.
    samples_train = SAMPLES[i](N_SAMPLES, rng)
    samples_test = SAMPLES[i](N_SAMPLES, rng)
    # Univariate KDE.
    kde = MultivariateKde([0], None, distargs={O: {ST: [N], SA:[{}]}}, rng=rng)
    # Incorporate observations.
    for rowid, x in enumerate(samples_train):
        kde.incorporate(rowid, {0: x})
    # Run inference.
    kde.transition()
    # Generate posterior samples.
    samples_gen = [s[0] for s in kde.simulate(-1, [0], N=N_SAMPLES)]
    # Plot comparison of all train, test, and generated samples.
    fig, ax = plt.subplots()
    ax.scatter(samples_train, [0]*len(samples_train), color='b', label='Train')
    ax.scatter(samples_gen, [1]*len(samples_gen), color='r', label='Posterior')
    ax.scatter(samples_test, [2]*len(samples_test), color='g', label='Test')
    ax.grid()
    plt.close()
    # KS test
    assert .01 < ks_2samp(samples_test, samples_gen)
