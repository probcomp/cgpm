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

"""This graphical test trains a gpmcc state on a bivariate population [X, Z].
X (called the data) is a cctype from DistributionGpm. Z is a categorical
variable that is a function of the latent cluster of each row
(called the indicator).

The three simulations are:
    - Joint Z,X.
    - Data conditioned on the indicator Z|X.
    - Indicator conditioned on the data X|Z.

Simulations are compared to synthetic data at indicator subpopulations.
"""

import pytest

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import ks_2samp

import gpmcc.utils.general as gu
import gpmcc.utils.test as tu

from gpmcc.crosscat.engine import Engine



N_SAMPLES = 250

T, Zv, Zc = tu.gen_data_table(
    N_SAMPLES, [1], [[.3, .5, .2]], ['normal'], [None], [.95],
    rng=gu.gen_rng(0))

DATA = np.zeros((N_SAMPLES, 2))
DATA[:,0] = T[0]

INDICATORS = [0, 1, 2, 3, 4, 5]

counts = {0:0, 1:0, 2:0}
for i in xrange(N_SAMPLES):
    k = Zc[0][i]
    DATA[i,1] = 2*INDICATORS[k] + counts[k] % 2
    counts[k] += 1


@pytest.fixture(scope='module')
def state():
    # Create an engine.
    engine = Engine(
        DATA, cctypes=['normal', 'categorical'], distargs=[None, {'k':6}],
        num_states=4, rng=gu.gen_rng(2))
    engine.transition(N=15)
    marginals = engine.logpdf_score()
    ranking = np.argsort(marginals)[::-1]
    return engine.get_state(ranking[0])


def test_joint(state):
    # Simulate from the joint distribution of (x,i).
    joint_samples = state.simulate(-1, [0,1], N=N_SAMPLES)
    _, ax = plt.subplots()
    ax.set_title('Joint Simulation')
    for t in INDICATORS:
        # Plot original data.
        data_subpop = DATA[DATA[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data.
        joint_samples_subpop = joint_samples[joint_samples[:,1] == t]
        ax.scatter(
            joint_samples_subpop[:,1] + .25, joint_samples_subpop[:,0],
            color=gu.colors[t])
        # KS test.
        pvalue = ks_2samp(data_subpop[:,0], joint_samples_subpop[:,0])[1]
        assert 0.05 < pvalue
    ax.set_xlabel('Indicator')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_indicator(state):
    # Simulate from the conditional X|Z
    _, ax = plt.subplots()
    ax.set_title('Conditional Simulation Of Data X Given Indicator Z')
    for t in INDICATORS:
        # Plot original data.
        data_subpop = DATA[DATA[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data.
        conditional_samples_subpop = state.simulate(
            -1, [0], evidence=[(1,t)], N=len(data_subpop))
        ax.scatter(
            np.repeat(t, len(data_subpop)) + .25,
            conditional_samples_subpop[:,0], color=gu.colors[t])
        # KS test.
        pvalue = ks_2samp(
            data_subpop[:,0], conditional_samples_subpop[:,0])[1]
        assert 0.1 < pvalue
    ax.set_xlabel('Indicator')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_real(state):
    # Simulate from the conditional Z|X
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Conditional Simulation Of Indicator Z Given Data X')
    # Compute representative data sample for each dindicator.
    means = [np.mean(DATA[DATA[:,1]==i], axis=0)[0] for
        i in INDICATORS]
    for mean, indicator, ax in zip(means, INDICATORS, axes.ravel('F')):
        conditional_samples_subpop = state.simulate(
            -1, [1], evidence=[(0,mean)], N=N_SAMPLES)
        ax.hist(conditional_samples_subpop, color='g', alpha=.4)
        ax.set_title('True Indicator %d' % indicator)
        ax.set_xlabel('Simulated Indicator')
        ax.set_xticks(INDICATORS)
        ax.set_ylabel('Frequency')
        ax.set_ylim([0, ax.get_ylim()[1]+10])
        ax.grid()
