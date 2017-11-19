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

from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu
from cgpm.utils import test as tu


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
        num_states=4, rng=gu.gen_rng(212))
    engine.transition(N=15)
    marginals = engine.logpdf_score()
    ranking = np.argsort(marginals)[::-1]
    return engine.get_state(ranking[0])


def test_joint(state):
    # Simulate from the joint distribution of (x,z).
    joint_samples = state.simulate(-1, [0,1], N=N_SAMPLES)
    _, ax = plt.subplots()
    ax.set_title('Joint Simulation')
    for t in INDICATORS:
        # Plot original data.
        data_subpop = DATA[DATA[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data for indicator t.
        samples_subpop = [j[0] for j in joint_samples if j[1] == t]
        ax.scatter(
            np.add([t]*len(samples_subpop), .25), samples_subpop,
            color=gu.colors[t])
        # KS test.
        pvalue = ks_2samp(data_subpop[:,0], samples_subpop)[1]
        assert .05 < pvalue
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
        samples_subpop = [s[0] for s in
            state.simulate(-1, [0], {1:t}, None, len(data_subpop))]
        ax.scatter(
            np.repeat(t, len(data_subpop)) + .25,
            samples_subpop, color=gu.colors[t])
        # KS test.
        pvalue = ks_2samp(data_subpop[:,0], samples_subpop)[1]
        assert .01 < pvalue
    ax.set_xlabel('Indicator')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_real(state):
    # Simulate from the conditional Z|X
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Conditional Simulation Of Indicator Z Given Data X')
    # Compute representative data sample for each indicator.
    means = [np.mean(DATA[DATA[:,1]==t], axis=0)[0] for t in INDICATORS]
    for mean, indicator, ax in zip(means, INDICATORS, axes.ravel('F')):
        samples_subpop = [s[1] for s in
            state.simulate(-1, [1], {0:mean}, None, N_SAMPLES)]
        ax.hist(samples_subpop, color='g', alpha=.4)
        ax.set_title('True Indicator %d' % indicator)
        ax.set_xlabel('Simulated Indicator')
        ax.set_xticks(INDICATORS)
        ax.set_ylabel('Frequency')
        ax.set_ylim([0, ax.get_ylim()[1]+10])
        ax.grid()
        # Check that the simulated indicator agrees with true indicator.
        true_ind_a = indicator
        true_ind_b = indicator-1  if indicator % 2 else indicator+1
        counts = np.bincount(samples_subpop)
        frac = sum(counts[[true_ind_a, true_ind_b]])/float(sum(counts))
        assert .8 < frac
