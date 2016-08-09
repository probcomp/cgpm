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

"""This tests the simulate methods of all the unconditional GPMs by observing
data from a mixture, learning, and comparing the posterior predictives.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scipy.stats import chisquare
from scipy.stats import ks_2samp

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


NUM_ITERS = 1000
NUM_SECONDS = 180

VIEW_WEIGHTS = [1]
CLUSTER_WEIGHTS = [[.3, .5, .2]]
SEPARATION = [.95]

NUM_TRAIN = 250
NUM_TEST = 450


def simulate_synthetic(n_samples, cctype, distargs):
    rng = gu.gen_rng(12)
    D, Zv, Zc = tu.gen_data_table(
        n_samples, VIEW_WEIGHTS, CLUSTER_WEIGHTS, [cctype], [distargs],
        SEPARATION, rng=rng)
    rng.shuffle(D[0])
    return np.asarray(D).T


def aligned_bincount(arrays):
    bincounts = [np.bincount(a.astype(int)) for a in arrays]
    longest = max(len(b) for b in bincounts)
    return [np.append(b, np.zeros(longest-len(b))) for b in bincounts]


def two_sample_test(cctype, X, Y):
    model = cu.cctype_class(cctype)
    if model.is_numeric(): # XXX WRONG CHOICE FOR DISCRETE NUMERIC XXX
        _, pval = ks_2samp(X, Y)
    else:
        Xb, Yb = aligned_bincount([X, Y])
        ignore = np.logical_and(Xb==0, Yb==0)
        Xb, Yb = Xb[np.logical_not(ignore)], Yb[np.logical_not(ignore)]
        Xb = Xb/float(sum(Xb)) * 1000
        Yb = Yb/float(sum(Yb)) * 1000
        _, pval = chisquare(Yb, f_exp=Xb)
    return pval


def plot_simulations(cctype, D_train, D_test, D_posterior):
    model = cu.cctype_class(cctype)
    if model.is_continuous():
        fig, ax = _plot_simulations_continuous(D_train, D_test, D_posterior)
    else:
        fig, ax = _plot_simulations_discrete(D_train, D_test, D_posterior)
    fig.suptitle(cctype, fontsize=16, fontweight='bold')
    fig.set_size_inches(8, 6)
    fig.savefig('resources/%s-%s' % (cu.timestamp(), cctype), dpi=100)
    plt.close('all')


def _plot_simulations_continuous(D_train, D_test, D_posterior):
    fig, ax = plt.subplots()

    def plot_samples(X, height=0, ax=None):
        if ax is None:
            _, ax = plt.subplots()
            ax.set_ylim([0, 10])
        for x in X:
            ax.vlines(x, height, height+1, linewidth=1)
        return ax

    samples = [D_train, D_test, D_posterior]
    colors = ['b', 'g', 'r']
    labels = ['In-Sample Data', 'Out-of-Sample Data', 'Posterior Simulations']
    for i, (D, c, l) in enumerate(zip(samples, colors, labels)):
        for x in D[:-1]:
            ax.vlines(x, 2*i, 2*i+.5, linewidth=1, color=c)
        ax.vlines(D[-1], 2*i, 2*i+.5, linewidth=1, color=c, label=l)
    ax.set_ylim([0, 10])
    ax.grid()
    ax.legend(framealpha=0)
    return fig, ax


def _plot_simulations_discrete(D_train, D_test, D_posterior):

    def align_axes(axes, xax=True):
        if xax:
            get_limits = lambda ax, i: ax.get_xlim()[i]
            set_limits = lambda ax, low, high: ax.set_xlim([low, high])
        else:
            get_limits = lambda ax, i: ax.get_ylim()[i]
            set_limits = lambda ax, low, high: ax.set_ylim([low, high])
        low = min([get_limits(ax, 0) for ax in axes])
        high = max([get_limits(ax, 1) for ax in axes])
        for ax in axes:
            set_limits(ax, low, high)
        return axes

    def histogram_axis(ax, D, c, l):
        freq = D / float(np.sum(D))
        ax.bar(xrange(len(D)), freq, color=c, label=l)
        ax.grid()
        ax.legend(framealpha=0)

    fig, axes = plt.subplots(3,1)

    samples = aligned_bincount([D_train, D_test, D_posterior])
    colors = ['b', 'g', 'r']
    labels = ['In-Sample Data', 'Out-of-Sample Data', 'Posterior Simulations']
    for (ax, D, c, l) in zip(axes, samples, colors, labels):
        histogram_axis(ax, D, c, l)
        ax.set_yticks(np.linspace(0, 1, 9))

    return fig, axes


def launch_2samp_sanity_same(cctype, distargs):
    """Ensure that 2-sample tests on same population does not reject H0."""
    # Training and test samples
    D = simulate_synthetic(NUM_TRAIN+NUM_TEST, cctype, distargs)
    D_train, D_test = np.ravel(D[:NUM_TRAIN]), np.ravel(D[NUM_TRAIN:])
    pval = two_sample_test(cctype, D_test, D_train)
    print 'cctype, pval: {}, {}'.format(cctype, pval)
    assert pval > 0.05


def launch_2samp_sanity_diff(cctype, distargs):
    """Ensure that 2-sample tests on different population rejects H0."""
    # Training and test samples
    D = simulate_synthetic(NUM_TRAIN+NUM_TEST, cctype, distargs)
    D_train = D[:NUM_TRAIN]
    D_posteriors = generate_gpmcc_posteriors(
        cctype, distargs, D_train, None, 0.01)
    pvals = [two_sample_test(cctype, np.ravel(D_train), np.ravel(Dp))
        for Dp in D_posteriors]
    print 'cctype, pvals: {}, {}'.format(cctype, pvals)
    assert all(p < 0.01 for p in pvals)


def launch_2samp_inference_quality(cctype, distargs):
    """Performs check of posterior predictive on gpmcc simulations."""
    # Training and test samples
    D = simulate_synthetic(NUM_TRAIN+NUM_TEST, cctype, distargs)
    D_train, D_test = D[:NUM_TRAIN], D[NUM_TRAIN:]
    D_posteriors = generate_gpmcc_posteriors(
        cctype, distargs, D_train, NUM_ITERS, NUM_SECONDS)
    for Dp in D_posteriors:
        plot_simulations(
            cctype, np.ravel(D_train), np.ravel(D_test[:NUM_TRAIN]),
            np.ravel(Dp))
    pvals = [two_sample_test(cctype, np.ravel(D_test), np.ravel(Dp))
        for Dp in D_posteriors]
    print 'cctype, pvals: {}, {}'.format(cctype, pvals)
    assert any([p > 0.05 for p in pvals])


def generate_gpmcc_posteriors(cctype, distargs, D_train, iters, seconds):
    """Learns gpmcc on D_train for seconds and simulates NUM_TEST times."""
    # Learning and posterior simulation.
    engine = Engine(
        D_train, cctypes=[cctype], distargs=[distargs],
        num_states=64, rng=gu.gen_rng(1))
    engine.transition(N=iters, S=seconds, progress=0)
    if iters:
        kernel = 'column_params' if cu.cctype_class(cctype).is_conditional()\
            else 'column_hypers'
        engine.transition(N=100, kernels=[kernel], progress=0)
    samples = engine.simulate(-1, [0], N=NUM_TEST)
    marginals = engine.logpdf_score()
    ranking = np.argsort(marginals)[::-1]
    for r in ranking[:5]:
        engine.get_state(r).plot()
    return [samples[i] for i in ranking[:5]]


cctypes_distargs = {
    'bernoulli'         : None,
    'beta'           : None,
    'categorical'       : {'k':12},
    'exponential'       : None,
    'geometric'         : None,
    'lognormal'         : None,
    'normal'            : None,
    'normal_trunc'      : {'l':0, 'h':50},
    'poisson'           : None,
    'vonmises'          : None,
}


@pytest.mark.parametrize('cctype', cctypes_distargs.keys())
def disabled_test_2samp_sanity_same(cctype):
    launch_2samp_sanity_same(cctype, cctypes_distargs[cctype])


@pytest.mark.parametrize('cctype', cctypes_distargs.keys())
def disabled_test_2samp_sanity_diff(cctype):
    launch_2samp_sanity_diff(cctype, cctypes_distargs[cctype])


@pytest.mark.parametrize('cctype', cctypes_distargs.keys())
def disabled_test_2samp_inference_quality__ci_(cctype):
    launch_2samp_inference_quality(cctype, cctypes_distargs[cctype])
