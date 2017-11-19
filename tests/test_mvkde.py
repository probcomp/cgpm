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
import pytest

from scipy.stats import chisquare
from scipy.stats import ks_2samp

from cgpm.kde.mvkde import MultivariateKde
from cgpm.uncorrelated.linear import Linear
from cgpm.utils import general as gu
from cgpm.utils import test as tu


O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'nominal'


def test_initialize():
    # This test ensures that MvKde raises on bad initialize arguments.
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
    # This test ensures that MvKde raises on bad incorporate.
    # No evidence.
    kde = MultivariateKde(
        outputs=[0, 1], inputs=None,
        distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Missing query.
    with pytest.raises(ValueError):
        kde.incorporate(0, {})
    # Disallow inputs.
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

@pytest.mark.parametrize('i', xrange(len(SAMPLES)))
def test_univariate_two_sample(i):
    # This test ensures posterior sampling of uni/bimodal dists on R. When the
    # plot is shown, a density curve overlays the samples which is useful for
    # seeing that logpdf/simulate agree.
    N_SAMPLES = 100

    rng = gu.gen_rng(2)
    # Synthetic samples.
    samples_train = SAMPLES[i](N_SAMPLES, rng)
    samples_test = SAMPLES[i](N_SAMPLES, rng)
    # Univariate KDE.
    kde = MultivariateKde([3], None, distargs={O: {ST: [N], SA:[{}]}}, rng=rng)
    # Incorporate observations.
    for rowid, x in enumerate(samples_train):
        kde.incorporate(rowid, {3: x})
    # Run inference.
    kde.transition()
    # Generate posterior samples.
    samples_gen = [s[3] for s in kde.simulate(-1, [3], N=N_SAMPLES)]
    # Plot comparison of all train, test, and generated samples.
    fig, ax = plt.subplots()
    ax.scatter(samples_train, [0]*len(samples_train), color='b', label='Train')
    ax.scatter(samples_gen, [1]*len(samples_gen), color='r', label='KDE')
    ax.scatter(samples_test, [2]*len(samples_test), color='g', label='Test')
    # Overlay the density function.
    xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
    pdfs = [kde.logpdf(-1, {3: x}) for x in xs]
    # Convert the pdfs from the range to 1 to 1.5 by rescaling.
    pdfs_plot = np.exp(pdfs)+1
    pdfs_plot = (pdfs_plot/max(pdfs_plot)) * 1.5
    ax.plot(xs, pdfs_plot, color='k')
    # Clear up some labels.
    ax.set_title('Univariate KDE Posterior versus Generator')
    ax.set_xlabel('x')
    ax.set_yticklabels([])
    # Show the plot.
    ax.grid()
    plt.close()
    # KS test
    _, p = ks_2samp(samples_test, samples_gen)
    assert .05 < p


@pytest.mark.parametrize('noise', [.1, .3, .7])
def test_bivariate_conditional_two_sample(noise):
    # This test checks joint and conditional simulation of a bivarate normal
    # with (correlation 1-noise). The most informative use is plotting but
    # there is a numerical test for the conditional distributions.
    N_SAMPLES = 100

    rng = gu.gen_rng(2)
    # Synthetic samples.
    linear = Linear(outputs=[0,1], noise=noise, rng=rng)
    samples_train = np.asarray(
        [[s[0], s[1]] for s in linear.simulate(-1, [0,1], N=N_SAMPLES)])
    # Bivariate KDE.
    kde = MultivariateKde(
        [0,1], None, distargs={O: {ST: [N,N], SA:[{},{}]}}, rng=rng)
    # Incorporate observations.
    for rowid, x in enumerate(samples_train):
        kde.incorporate(rowid, {0: x[0], 1: x[1]})
    # Run inference.
    kde.transition()
    # Generate posterior samples from the joint.
    samples_gen = np.asarray(
        [[s[0],s[1]] for s in kde.simulate(-1, [0,1], N=N_SAMPLES)])
    # Plot comparisons of the joint.
    fig, ax = plt.subplots(nrows=1, ncols=2)
    plot_data = zip(
        ax, ['b', 'r'], ['Train', 'KDE'], [samples_train, samples_gen])
    for (a, c, l, s) in plot_data:
        a.scatter(s[:,0], s[:,1], color=c, label=l)
        a.grid()
        a.legend(framealpha=0)
    # Generate posterior samples from the conditional.
    xs = np.linspace(-3, 3, 100)
    cond_samples_a = np.asarray(
        [[s[1] for s in linear.simulate(-1, [1], {0: x0}, N=N_SAMPLES)]
        for x0 in xs])
    cond_samples_b = np.asarray(
        [[s[1] for s in kde.simulate(-1, [1], {0: x0}, N=N_SAMPLES)]
        for x0 in xs])
    # Plot the mean value on the same plots.
    for (a, s) in zip(ax, [cond_samples_a, cond_samples_b]):
        a.plot(xs, np.mean(s, axis=1), linewidth=3, color='g')
        a.set_xlim([-5,4])
        a.set_ylim([-5,4])
    plt.close('all')
    # Perform a two sample test on the means.
    mean_a = np.mean(cond_samples_a, axis=1)
    mean_b = np.mean(cond_samples_b, axis=1)
    _, p = ks_2samp(mean_a, mean_b)
    assert .01 < p


def test_univariate_categorical():
    # This test generates univariate data from a nominal variable with 6 levels
    # and probability vector p_theory, and performs a chi-square test on
    # posterior samples from MvKde.

    rng = gu.gen_rng(2)
    N_SAMPLES = 1000
    p_theory = [.3, .1, .2, .15, .15, .1]
    samples_test = rng.choice(range(6), p=p_theory, size=N_SAMPLES)
    kde = MultivariateKde(
        [7], None, distargs={O: {ST: [C], SA:[{'k': 6}]}}, rng=rng)
    # Incorporate observations.
    for rowid, x in enumerate(samples_test):
        kde.incorporate(rowid, {7: x})
    kde.transition()
    # Posterior samples.
    samples_gen = kde.simulate(-1, [7], N=N_SAMPLES)
    f_obs = np.bincount([s[7] for s in samples_gen])
    f_exp = np.bincount(samples_test)
    _, pval = chisquare(f_obs, f_exp)
    assert 0.05 < pval
    # Get some coverage on logpdf_score.
    assert kde.logpdf_score() < 0


def test_noisy_permutation_categorical():
    # This test builds a synthetic bivariate distribution for variables X and Y,
    # which are both categorical(3). The relationship is y = f(X) where f is
    # the permutation (0,1,2)->(1,2,0). To introduce noise, 10 percent of the
    # samples are "corrupted" and do not obey the relationship. The test ensure
    # posterior simulate/logpdf target the permutation, and agree with one
    # another.

    rng = gu.gen_rng(22)
    N_SAMPLES = 250

    f_permutation = {0:1, 1:2, 2:0}
    b_permutation = {0:2, 1:0, 2:1}

    X = rng.choice([0,1,2], p=[.33, .33, .34], size=N_SAMPLES).astype(float)
    Y = (X+1) % 3

    # Corrupt 10% of the samples.
    corruption = rng.choice(
        range(N_SAMPLES), replace=False, size=int(.1*N_SAMPLES))
    for c in corruption:
        Y[c] = rng.choice([i for i in f_permutation if i!=Y[c]])

    # Add 2 nans.
    X[0] = np.nan
    Y[4] = np.nan

    samples_test = np.column_stack((X,Y))

    # Build MvKde.
    kde = MultivariateKde(
        [7,8], None,
        distargs={O: {ST: [C, C], SA:[{'k': 3}, {'k': 3}]}}, rng=rng)
    for rowid, x in enumerate(samples_test):
        kde.incorporate(rowid, {7:x[0], 8:x[1]})
    kde.transition()

    def test_sample_match(s, target):
        f_obs = np.bincount(s)
        f_exp = [90 if i==target else 5 for i in [0,1,2]]
        # Max should be the target.
        amax_obs = np.argmax(f_obs)
        amax_exp = np.argmax(f_exp)
        assert amax_obs == amax_exp
        # Noise should not account for more than .20
        n_noise = sum(f for i,f in enumerate(f_obs) if i!=amax_obs)
        frac_noise = n_noise / float(sum(f_obs))
        assert frac_noise < 0.20

    def test_logps_match(s, ps):
        f_obs = np.bincount(s)
        n = float(sum(f_obs))
        p_obs = f_obs / n
        _, pval = chisquare(p_obs*n, ps*n)
        assert 0.05 < pval

    # Generate posterior samples conditioning on 7.
    for g in [0,1,2]:
        samples = [s[8] for s in kde.simulate(-1, [8], {7: g}, N=10000)]
        test_sample_match(samples, f_permutation[g])
        logps = [kde.logpdf(-1, {8:i}, {7:g}) for i in f_permutation]
        test_logps_match(samples, np.exp(logps))

    # Generate posterior samples conditioning on 8.
    for g in [0,1,2]:
        samples = [s[7] for s in kde.simulate(-1, [7], {8: g}, N=10000)]
        test_sample_match(samples, b_permutation[g])
        logps = [kde.logpdf(-1, {7:i}, {8:g}) for i in f_permutation]
        test_logps_match(samples, np.exp(logps))


def test_transition_no_data():
    kde = MultivariateKde(
        [1], None, distargs={O: {ST: [N], SA: [{}]}}, rng=gu.gen_rng(0))
    bw = list(kde.bw)
    kde.transition()
    assert np.allclose(bw, kde.bw)


def test_serialize():
    rng = gu.gen_rng(1)

    data = rng.rand(20, 5)
    data[:10,-1] = 0
    data[10:,-1] = 1

    kde = MultivariateKde(
        range(5), None,
        distargs={O: {ST: [N, N, N, N, C], SA: [{},{},{},{},{'k':1}]}}, rng=rng)
    for rowid, x in enumerate(data):
        kde.incorporate(rowid, dict(zip(range(5), x)))
    kde.transition()

    metadata_s = json.dumps(kde.to_metadata())
    metadata_l = json.loads(metadata_s)

    modname = importlib.import_module(metadata_l['factory'][0])
    builder = getattr(modname, metadata_l['factory'][1])
    kde2 = builder.from_metadata(metadata_l, rng=rng)

    # Variable indexes.
    assert kde2.outputs == kde.outputs
    assert kde2.inputs == kde.inputs
    # Distargs.
    assert kde2.get_distargs() == kde.get_distargs()
    # Dataset.
    assert kde2.data == kde.data
    assert kde2.N == kde.N
    # Bandwidth params.
    assert np.allclose(kde2.bw, kde.bw)
    # Statistical types.
    assert kde2.stattypes == kde.stattypes


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
def kde_xz():
    # Learns an MvKde on the dataset generated by generate_real_nominal_data
    # and returns the fixture for use in the next three tests.

    N_SAMPLES = 250
    data, indicators = generate_real_nominal_data(N_SAMPLES)
    K = MultivariateKde(
        [0,1], None,
        distargs={O: {ST: [N, C], SA:[{}, {'k': len(indicators)}]}},
        rng=gu.gen_rng(0))
    for rowid, x in enumerate(data):
        K.incorporate(rowid, {0:x[0], 1:x[1]})
    K.transition()
    return K


def test_joint(kde_xz):
    # Simulate from the joint distribution of x,z (see
    # generate_real_nominal_data) and perform a KS tests at each of the
    # subpopulations at the six levels of z.

    data = np.asarray(kde_xz.data.values())
    indicators = sorted(set(data[:,1].astype(int)))
    joint_samples = kde_xz.simulate(-1, [0,1], N=len(data))
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
        _, p = ks_2samp(data_subpop[:,0], samples_subpop)
        assert .05 < p
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_indicator(kde_xz):
    # Simulate from the conditional distribution of x|z (see
    # generate_real_nominal_data) and perfrom a KS tests at each of the
    # subpopulations at the six levels of z.

    data = np.asarray(kde_xz.data.values())
    indicators = sorted(set(data[:,1].astype(int)))
    _, ax = plt.subplots()
    ax.set_title('Conditional Simulation Of X Given Indicator Z')
    for t in indicators:
        # Plot original data.
        data_subpop = data[data[:,1] == t]
        ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
        # Plot simulated data.
        samples_subpop = [s[0] for s in
            kde_xz.simulate(-1, [0], {1:t}, None, N=len(data_subpop))]
        ax.scatter(
            np.repeat(t, len(data_subpop)) + .25,
            samples_subpop, color=gu.colors[t])
        # KS test.
        _, p = ks_2samp(data_subpop[:,0], samples_subpop)
        assert .1 < p
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.grid()


def test_conditional_real(kde_xz):
    # Simulate from the conditional distribution of z|x (see
    # generate_real_nominal_data) and plot the frequencies of the simulated
    # values.

    data = np.asarray(kde_xz.data.values())
    indicators = sorted(set(data[:,1].astype(int)))
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Conditional Simulation Of Indicator Z Given X', size=20)
    # Compute representative data sample for each indicator.
    means = [np.mean(data[data[:,1]==t], axis=0)[0] for t in indicators]
    for mean, indicator, ax in zip(means, indicators, axes.ravel('F')):
        samples_subpop = [s[1] for s in
            kde_xz.simulate(-1, [1], {0:mean}, None, N=len(data))]
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
