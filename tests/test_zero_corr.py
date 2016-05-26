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

import multiprocessing as mp

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

from bayeslite.math_util import logmeanexp

from gpmcc.utils.xy_gpm.diamond import DiamondGpm
from gpmcc.utils.xy_gpm.dots import DotsGpm
from gpmcc.utils.xy_gpm.linear import LinearGpm
from gpmcc.utils.xy_gpm.parabola import ParabolaGpm
from gpmcc.utils.xy_gpm.ring import RingGpm
from gpmcc.utils.xy_gpm.sin import SinGpm
from gpmcc.utils.xy_gpm.xcross import XCrossGpm

from gpmcc.engine import Engine
from gpmcc.utils import config as cu
from gpmcc.utils import entropy_estimators as ee
from gpmcc.utils.general import gen_rng


TIMESTAMP = cu.timestamp()
NUM_SAMPLES = 200
NUM_STATES = 12
NUM_SECONDS = 250
NOISE = [.95, .85, .75, .65, .55, .45, .35, .25, .15, .05, .01]


simulators = {
    'diamond': DiamondGpm,
    'dots': DotsGpm,
    'linear': LinearGpm,
    'parabola': ParabolaGpm,
    'ring': RingGpm,
    'sin': SinGpm,
    'xcross': XCrossGpm,
    }


simulator_limits = {
    'diamond': ([-1.5, 1.5],[-1.5, 1.5]),
    'dots': ([-2.5, 2.5],[-2.5, 2.5]),
    'linear': ([-3, 3],[-3, 3]),
    'parabola': ([-1.5, 1.5],[-1.5, 1.5]),
    'ring': ([-1.5, 1.5],[-1.5, 1.5]),
    'sin': ([-5, 5],[-2, 2]),
    'xcross': ([-3, 3],[-3, 3]),
    }


def filename_prefix(dist):
    return 'resources/%s/%s' % (TIMESTAMP, dist)

def filename_engine(dist, noise):
    return '%s/%s_%1.2f.pkl' % (filename_prefix(dist), dist, noise)

def filename_dataset(dist, noise):
    return '%s/dataset_%s_%1.2f.png' % (filename_prefix(dist), dist, noise)

def filename_samples(dist, noise):
    return '%s/samples_%s_%1.2f.txt' % (filename_prefix(dist), dist, noise)

def filename_mi(dist, noise):
    return '%s/mi_%s_%1.2f.txt' % (filename_prefix(dist), dist, noise)

def filename_samples_figure(dist, noise):
    return '%s/samples_%s_%1.2f.png' % (filename_prefix(dist), dist, noise)

def filename_mi_figure(dist):
    return '%s/mi_%s.png' % (filename_prefix(dist), dist)

def simulate_dataset(dist, noise, size=NUM_SAMPLES):
    rng = gen_rng(0)
    return simulators[dist](noise=noise, rng=rng).simulate_xy(size=size)

def create_engine(dist, noise):
    T = simulate_dataset(dist, noise)
    print 'Creating engine (%s %1.2f) ...' % (dist, noise)
    engine = Engine(T, cctypes=['normal','normal'], num_states=NUM_STATES)
    engine.to_pickle(file(filename_engine(dist, noise), 'w'))

def load_engine(dist, noise):
    print 'Loading %s %f' % (dist, noise)
    return Engine.from_pickle(file(filename_engine(dist, noise), 'r'))

def train_engine(dist, noise, S=NUM_SECONDS):
    print 'Transitioning engine (%s %1.2f) ...' % (dist, noise)
    engine = load_engine(dist, noise)
    engine.transition(S=S)
    engine.to_pickle(file(filename_engine(dist, noise), 'w'))

def load_engine_states(engine, top=None):
    """Load the top states from the engine by log score."""
    print 'Loading states'
    marginals = engine.logpdf_score()
    states = np.argsort(marginals)[::-1]
    states = [engine.get_state(i) for i in states[:top]]
    return states

def retrieve_nice_state(states, transitions=10):
    """Heuristic for selecting nice state to plot."""
    print 'Retrieving nice state from engine to plot'
    for s in states:
        if len(s.views) == 1:
            state = s
            break
    else:
        state = states[0]
    for _ in transitions(transitions):
        state.transition_column_hypers()
    return state

def plot_samples(samples, dist, noise):
    """Plot the observed samples and posterior samples side-by-side."""
    fig, ax = plt.subplots(nrows=1, ncols=2)
    # Plot the observed samples.
    T = simulate_dataset(dist, noise)
    print 'Plotting %s Distribution (Noise %1.2f)' % (dist, noise)
    ax[0].set_title('%s Distribution (Noise %1.2f)' % (dist, noise))
    ax[0].scatter(T[:,0], T[:,1], color='k', alpha=.5, label='True Distribution')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].grid()
    # Plot posterior distribution.
    ax[1].set_title('Emulator Distribution')
    ax[1].set_xlabel('x1')
    ax[1].grid()
    clusters = set(samples[:,2])
    colors = iter(cm.gist_rainbow(np.linspace(0, 1, len(clusters)+2)))
    for c in clusters:
        sc = samples[samples[:,2] == c][:,[0,1]]
        ax[1].scatter(sc[:,0], sc[:,1], alpha=.5, color=next(colors))
    ax[0].set_xlim(simulator_limits[dist][0])
    ax[0].set_ylim(simulator_limits[dist][1])
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    # Save.
    fig.savefig(filename_samples_figure(dist, noise))
    plt.close('all')

def plot_mi(mis, dist, noises):
    """Plot estimates of the mutual information by noise level."""
    fig, ax = plt.subplots()
    ax.set_title('Mutual Information (%s) ' % dist, fontweight='bold')
    ax.set_xlabel('Noise', fontweight='bold')
    ax.set_ylabel('Mutual Information (Estimates)', fontweight='bold')
    ax.plot(noises, np.mean(mis, axis=1), color='red', label='Emulator Mean')
    for i, row in enumerate(mis.T):
        label = 'Emulator' if i == len(mis.T) - 1 else None
        ax.scatter(noises, row, alpha=.2, color='green', label=label)
    ax.grid()
    ax.legend(framealpha=0)
    # Save.
    fig.savefig(filename_mi_figure(dist))
    plt.close('all')


# dist = 'diamond'
# dist = 'dots'
# dist = 'linear'
# dist = 'parabola'
# dist = 'ring'
# dist = 'sin'
# dist = 'xcross'


def launch_analysis(dist):
    # Train gpmcc engines.
    for noise in NOISE:
        create_engine(dist, noise)
        train_engine(dist, noise)


def generate_samples(dist, noise):
    engine = load_engine(dist, noise)
    state = retrieve_nice_state(load_engine_states(engine))
    samples = state.views[0]._simulate_hypothetical(
        [0,1], [], NUM_SAMPLES, cluster=True)
    np.savetxt(filename_samples(dist, noise), samples, delimiter=',')


def generate_mi(dist, noise):
    engine = load_engine(dist, noise)
    mi = engine.mutual_information(0, 1)
    np.savetxt(filename_samples(dist, noise), [mi], delimiter=',')


def plot_samples_all(dist):
    for noise in NOISE:
        samples  = np.loadtxt(
            filename_samples(dist, noise), delimiter=',')
        plot_samples(samples, dist, noise)


def plot_mi_all(dist):
    filenames = [filename_mi(dist, noise) for noise in NOISE]
    mis = [float(np.loadtxt(f, delimiter=',')) for f in filenames]
    plot_mi(mis, dist, NOISE)
