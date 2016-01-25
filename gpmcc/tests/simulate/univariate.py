# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This test suite targets the posterior inference and `simulate` method of
the `DistirbutionGpm` classes. The test approach is

1. Generate a univariate dataset X from a synthetic mixture with assignments Z.

2. Train gpmcc on the full data [X, Z] where Z is a categorical column that is
a deterministic function of the assignments (in the simplest case, Z are the
exact assignments) for some number of

3. Simulate from the joint posterior [X', Z'] ~ DistributionGpm(|[X,Z]).
- Simulate more synthetic data [X'', Z''] from the synthetic generator in 1.

4. Perform a two-sample test comparing [X', Z'] vs [X'', Z''] in a variety of
ways, such as
    - Testing the marginals by just comparing X' and X''.
    - Testing at the conditions by comparing X', X'' for subpopulations based
    on the simulated values of Z.
    - Other possibilities.
"""

import numpy as np

import gpmcc.utils.sampling as su
import gpmcc.utils.test as tu
from gpmcc.engine import Engine

# Sample parameters.
n_samples = 50
cluster_weights = [.15, .35, .25, .25, .1]
indicators = [[k, k+1] for k in xrange(0, 2*len(cluster_weights), 2)]
indicator_distargs = {'k': 2*len(indicators)}
separation = [.8]

# Analysis paramters.
num_states = 1
num_iter = 10

def compute_indicators(indicators, Zr):
    """Convert assignments into categorical indicators which creates a 2 to
    1 mapping."""
    assert len(indicators) == len(set(Zr))
    Zr_prime = np.zeros(len(Zr))
    for i, z in enumerate(Zr):
        Zr_prime[i] = np.random.choice(indicators[z])
    return Zr_prime

def generate_gpmcc_state(cctype, distargs=None):
    # Obtain synthetic data.
    D = np.zeros((n_samples, 2))
    D[:,0], _, Zc = tu.gen_data_table(n_samples, [1], [cluster_weights],
        [cctype], [distargs], separation)
    D[:,1] = compute_indicators(indicators, Zc[0])

    # Create an engine and analyze.
    state = Engine(D, [cctype, 'categorical'],
        [distargs, indicator_distargs], num_states=num_states,
        initialize=True)
    state.transition(N=num_iter)
    return D, state.get_state(0)

def continuous_two_sample_test(D, Q):
    """D and Q are lists of floats to perform a two-sample test."""

def discrete_two_sample_test(D, Q):
    """D and Q are lists of integers to perform a two-sample test."""

def test_bernoulli():
    D, model= generate_gpmcc_state('bernoulli')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, samples

def test_beta_uc():
    D, model= generate_gpmcc_state('beta_uc')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, samples

def test_categorical():
    D, model= generate_gpmcc_state('categorical', distargs=indicator_distargs)
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_exponential():
    D, model= generate_gpmcc_state('exponential')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_exponential_uc():
    D, model= generate_gpmcc_state('exponential_uc')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_geometric():
    D, model= generate_gpmcc_state('geometric')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_lognormal():
    D, model= generate_gpmcc_state('lognormal')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_normal():
    D, model= generate_gpmcc_state('normal')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_poisson():
    D, model= generate_gpmcc_state('poisson')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples

def test_vonmises():
    D, model= generate_gpmcc_state('vonmises')
    samples = model.simulate(None, [0, 1], N=n_samples)
    return D, model, samples
