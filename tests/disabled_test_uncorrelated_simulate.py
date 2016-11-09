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

'''
Tests simulate from the joint distribution of the XY CGPMs in directory
uncorrelated/. The suite is disabled because currently only plots of the
synthetic data are generated, and it is yet to be determined how to test the
plots.

Moreover a wider set of test cases for _conditional_ simulate are required.
'''

import itertools
import pytest

import matplotlib.pyplot as plt
import numpy as np

from cgpm.uncorrelated.diamond import Diamond
from cgpm.uncorrelated.dots import Dots
from cgpm.uncorrelated.linear import Linear
from cgpm.uncorrelated.parabola import Parabola
from cgpm.uncorrelated.ring import Ring
from cgpm.uncorrelated.sin import Sin
from cgpm.uncorrelated.xcross import XCross

from cgpm.utils.general import gen_rng


NUM_SAMPLES = 200
NOISES = [.95, .85, .75, .65, .55, .45, .35, .25, .15, 0.10, .05, .01]

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


# --------------------------------------------------------------------------
# Inference.

def simulate_dataset(dist, noise, size=200):
    rng = gen_rng(0)
    cgpm = simulators[dist](outputs=[2,4], noise=noise, rng=rng)
    samples = cgpm.simulate(-1, [2, 4], N=size)
    try: # XXX Crash test only!
        logpdfs = [cgpm.logpdf(-1, s) for s in samples]
    except NotImplementedError:
        pass
    D = [(s[2], s[4]) for s in samples]
    return np.asarray(D)

def plot_synthetic(dist, noise, size=1000):
    fig, ax = plt.subplots()
    T = simulate_dataset(dist, noise, size=size)
    ax.set_title('%s Distribution (Noise %1.2f)' % (dist, noise))
    ax.scatter(T[:,0], T[:,1], color='k', alpha=.5, label='True Distribution')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid()
    ax.set_xlim(simulator_limits[dist][0])
    ax.set_ylim(simulator_limits[dist][1])
    fig.savefig('/tmp/synth_%s_%1.2f.png' % (dist, noise))
    plt.close('all')

@pytest.mark.parametrize('dist, noise', itertools.product(simulators, NOISES))
def test_dist_noise__ci_(dist, noise):
    plot_synthetic(dist, noise)
