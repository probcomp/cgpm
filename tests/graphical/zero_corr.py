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

"""
This script produces the following directory structure

|- zero_corr.py
|- resources/
    |-timetsamp#1/
        |- diamond/
            |- samples_diamond_0.01.png
            |- samples_diamond_0.05.png
            ...
            |- samples_diamond_0.95.png
            |- mi_diamond.png
        |- parabola/
        ...
        |- ring/
    |-timestamp#2/
        |...

To invoke
    `python zero_corr.py`
which will launch all the jobs in parallel and wait for them to terminate, then
produce the plots. Note that the script uses all available processors so will
cause heavy overload on the machine.

For each distribution (diamond, sin, ring, etc) two plots can be generated:

    1. Plot of samples versus true distribution (samples_<dist>_<noise>].png)
        which is 11 plots, that can either be organized into a single PNG or
        shown in sequence.

    2. Plot of the mutual information versus noise (mi_<dist>.png) which is 1
        plot showing the mutual information varying with noise level.

The estimate runtime for the whole script with the current configuration is

    30 min per dist, (10 min per noise, 64/16=4 in parallel, 12/4=3 chunks)
    7 total dists = 210 min
"""

import argparse
import itertools
import os
import subprocess

from datetime import datetime

import matplotlib
matplotlib.use('Agg')

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

from cgpm.uncorrelated.diamond import Diamond
from cgpm.uncorrelated.dots import Dots
from cgpm.uncorrelated.linear import Linear
from cgpm.uncorrelated.parabola import Parabola
from cgpm.uncorrelated.ring import Ring
from cgpm.uncorrelated.sin import Sin
from cgpm.uncorrelated.xcross import XCross

from cgpm.crosscat.engine import Engine
from cgpm.utils import config as cu
from cgpm.utils import entropy_estimators as ee
from cgpm.utils.general import gen_rng

# --------------------------------------------------------------------------
# Configuration.

NUM_SAMPLES = 200
NUM_STATES = 16
NUM_SECONDS = 600
NOISES = [.95, .85, .75, .65, .55, .45, .35, .25, .15, .10, .05, .01]

NUM_PROCESSORS = 64
NUM_PARALLEL = NUM_PROCESSORS/NUM_STATES
SLICES = range(0, len(NOISES), NUM_PARALLEL) + [None]
CKPTS = [NOISES[SLICES[i-1]:SLICES[i]] for i in xrange(1, len(SLICES))]


def get_latest_timestamp():
    def get_datetime(timestamp):
        try:
            return datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
        except ValueError:
            return None
    timestamps = os.listdir('./resources')
    dates = [get_datetime(t) for t in timestamps]
    highest = sorted(d for d in dates if d is not None)[-1]
    return highest.strftime('%Y%m%d-%H%M%S')


simulators = {
    'diamond': Diamond,
    'dots': Dots,
    'linear': Linear,
    'parabola': Parabola,
    'ring': Ring,
    'sin': Sin,
    'xcross': XCross,
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


def filename_prefix(dist, timestamp):
    return 'resources/%s/%s' % (timestamp, dist)

def filename_engine(dist, noise, timestamp):
    return '%s/%s_%1.2f.pkl'\
        % (filename_prefix(dist, timestamp), dist, noise)

def filename_dataset(dist, noise, timestamp):
    return '%s/dataset_%s_%1.2f.png'\
        % (filename_prefix(dist, timestamp), dist, noise)

def filename_samples(dist, noise, modelno, timestamp):
    return '%s/samples_%s_%1.2f_%d.txt'\
        % (filename_prefix(dist, timestamp), dist, noise, modelno)

def filename_mi(dist, noise, timestamp):
    return '%s/mi_%s_%1.2f.txt'\
        % (filename_prefix(dist, timestamp), dist, noise)

def filename_samples_figure(dist, noise, modelno, timestamp):
    return '%s/samples_%s_%1.2f_%d.png'\
        % (filename_prefix(dist, timestamp), dist, noise, modelno)

def filename_mi_figure(dist, timestamp):
    return '%s/mi_%s.png' % (filename_prefix(dist, timestamp), dist)

# --------------------------------------------------------------------------
# Inference.

def simulate_dataset(dist, noise, size=200):
    rng = gen_rng(100)
    cgpm = simulators[dist](outputs=[0,1], noise=noise, rng=rng)
    samples = [cgpm.simulate(-1, [0, 1]) for i in xrange(size)]
    D = [(s[0], s[1]) for s in samples]
    return np.asarray(D)

def create_engine(dist, noise, num_samples, num_states, timestamp):
    T = simulate_dataset(dist, noise, num_samples)
    print 'Creating engine (%s %1.2f) ...' % (dist, noise)
    engine = Engine(T, cctypes=['normal','normal'], num_states=num_states)
    engine.to_pickle(file(filename_engine(dist, noise, timestamp), 'w'))

def load_engine(dist, noise, timestamp):
    print 'Loading %s %f' % (dist, noise)
    return Engine.from_pickle(file(filename_engine(dist, noise, timestamp),'r'))

def train_engine(dist, noise, seconds, timestamp):
    print 'Transitioning (%s %1.2f) ...' % (dist, noise)
    engine = load_engine(dist, noise, timestamp)
    engine.transition(S=seconds)
    engine.to_pickle(file(filename_engine(dist, noise, timestamp), 'w'))

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
    for _ in xrange(transitions):
        state.transition_dim_hypers()
    return state

def plot_samples(samples, dist, noise, modelno, num_samples, timestamp):
    """Plot the observed samples and posterior samples side-by-side."""
    print 'Plotting samples %s %f' % (dist, noise)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(
        '%s (noise %1.2f, sample %d)' % (dist, noise, modelno),
        size=16)
    # Plot the observed samples.
    T = simulate_dataset(dist, noise, num_samples)
    # ax[0].set_title('Observed Data')
    ax[0].text(
        .5, .95, 'Observed Data',
        horizontalalignment='center',
        transform=ax[0].transAxes)
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].scatter(T[:,0], T[:,1], color='k', alpha=.5)
    ax[0].set_xlim(simulator_limits[dist][0])
    ax[0].set_ylim(simulator_limits[dist][1])
    ax[0].grid()
    # Plot posterior distribution.
    # ax[1].set_title('CrossCat Posterior Samples')
    ax[1].text(
        .5, .95, 'CrossCat Posterior Samples',
        horizontalalignment='center',
        transform=ax[1].transAxes)
    ax[1].set_xlabel('x1')
    clusters = set(samples[:,2])
    colors = iter(matplotlib.cm.gist_rainbow(
        np.linspace(0, 1, len(clusters)+2)))
    for c in clusters:
        sc = samples[samples[:,2] == c][:,[0,1]]
        ax[1].scatter(sc[:,0], sc[:,1], alpha=.5, color=next(colors))
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].grid()
    # Save.
    # fig.set_tight_layout(True)
    fig.savefig(filename_samples_figure(dist, noise, modelno, timestamp))
    plt.close('all')

def plot_mi(mis, dist, noises, timestamp):
    """Plot estimates of the mutual information by noise level."""
    print 'Plotting mi %s' % (dist)
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
    fig.savefig(filename_mi_figure(dist, timestamp))
    plt.close('all')


# --------------------------------------------------------------------------
# Launchers for dist, noise.

def generate_engine(dist, noise, num_samples, num_states, num_seconds, timestamp):
    create_engine(dist, noise, num_samples, num_states, timestamp)
    train_engine(dist, noise, num_seconds, timestamp)

def generate_samples(dist, noise, num_samples, timestamp):
    print 'Generating samples %s %f' % (dist, noise)
    engine = load_engine(dist, noise, timestamp)
    for modelno in xrange(NUM_STATES):
        state = engine.get_state(modelno)
        if len(state.views) == 1:
            for _ in xrange(10):
                state.transition_dim_hypers()
            view = state.view_for(0)
            sims = view.simulate(-1, [0, 1, view.outputs[0]], N=num_samples)
            samples = [[s[0],s[1],s[view.outputs[0]]] for s in sims]
        else:
            sims = state.simulate(-1, [0, 1], N=num_samples)
            samples = [[s[0],s[1],0] for s in sims]
        np.savetxt(
            filename_samples(dist, noise, modelno, timestamp),
            samples,
            delimiter=',')

def generate_mi(dist, noise, timestamp):
    print 'Generating mi %s %f' % (dist, noise)
    engine = load_engine(dist, noise, timestamp)
    mi = engine.mutual_information([0], [1])
    np.savetxt(filename_mi(dist, noise, timestamp), [mi], delimiter=',')

def plot_samples_all(dist, noises, modelnos, num_samples, timestamp):
    for noise, modelno in itertools.product(noises, modelnos):
        samples  = np.loadtxt(
            filename_samples(dist, noise, modelno, timestamp),
            delimiter=',')
        plot_samples(samples, dist, noise, modelno, num_samples, timestamp)

def plot_mi_all(dist, noises, timestamp):
    filenames = [filename_mi(dist, noise, timestamp) for noise in noises]
    mis = np.asarray([np.loadtxt(f, delimiter=',') for f in filenames])
    plot_mi(mis, dist, noises, timestamp)


# --------------------------------------------------------------------------
# Launchers for dist.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--distribution', type=str, help='Name of distribution to run.')
    parser.add_argument(
        '-n', '--noise', type=float, help='Noise level of distribution.')
    parser.add_argument(
        '-t', '--timestamp', type=str, help='Timestamp of the experiment.')
    args = parser.parse_args()

    # Creates the structure for this run.
    def create_directories(timestamp):
        if 'resources' not in os.listdir('.'):
            os.mkdir('./resources')
        os.mkdir('./resources/%s' % timestamp)
        for dist in simulators:
            os.mkdir('./resources/%s/%s' % (timestamp, dist))

    # Runs a single (dist, noise) experiment.
    def run_dist_noise(dist, noise, timestamp):
        generate_engine(
            dist, noise, NUM_SAMPLES, NUM_STATES, NUM_SECONDS, timestamp)
        generate_samples(dist, noise, NUM_SAMPLES, timestamp)
        generate_mi(dist, noise, timestamp)

    def run_dist(dist, timestamp):
        # Run these in parallel.
        for ckpt in CKPTS:
            processes = []
            for n in ckpt:
                proc = subprocess.Popen(
                    ['python ./zero_corr.py -d %s -n %1.2f -t %s'
                        % (dist, n, timestamp)],
                    shell=True)
                processes.append(proc)
            for p in processes:
                p.wait()
        return None

    # If subprocess, just launch run_dist_noise, otherwise run everything.
    if args.distribution is not None:
        assert args.noise is not None
        assert args.timestamp is not None
        run_dist_noise(args.distribution, args.noise, args.timestamp)
    else:
        tstamp = cu.timestamp()
        create_directories(tstamp)
        for d in simulators:
            run_dist(d, tstamp)
            plot_samples_all(d, NOISES, range(NUM_STATES), NUM_SAMPLES, tstamp)
            plot_mi_all(d, NOISES, tstamp)
