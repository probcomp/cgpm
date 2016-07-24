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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scipy.stats import ks_2samp
from scipy.stats import chisquare

from cgpm.kde.mvkde import MultivariateKde
from cgpm.utils import general as gu
from cgpm.utils import test as tu
from cgpm.uncorrelated.linear import Linear


O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'categorical'


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
    # This test ensures posterior sampling of uni/bimodal dists on R. When the
    # plot is shown, a density curve overlays the samples which is useful for
    # seeing that logpdf/simulate agree.

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
    assert .05 < ks_2samp(samples_test, samples_gen).pvalue


@pytest.mark.parametrize('noise', [.1, .3, .7])
def test_bivariate_conditional_two_sample(noise):
    # This test checks joint and conditional simulation of a bivarate normal
    # with (correlation 1-noise). The most informative use is plotting but
    # there is a numerical test for the conditional distributions.

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
    plt.close('all')
    # Perform a two sample test on the means.
    mean_a = np.mean(cond_samples_a, axis=1)
    mean_b = np.mean(cond_samples_b, axis=1)
    assert .01 < ks_2samp(mean_a, mean_b).pvalue


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


def test_noisy_permutation_categorical__ci_():
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

    X = rng.choice([0,1,2], p=[.33, .33, .34], size=N_SAMPLES)
    Y = (X+1) % 3

    # Corrupt 10% of the samples.
    corruption = rng.choice(
        range(N_SAMPLES), replace=False, size=int(.1*N_SAMPLES))
    for c in corruption:
        Y[c] = rng.choice([i for i in f_permutation if i!=Y[c]])

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
