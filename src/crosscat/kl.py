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

import numpy as np
import pytest

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.engine import _evaluate as _engine_evaluate

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu
from cgpm.utils.parallel_map import parallel_map


def compute_pairwise_kl(engine_0, engine_1, num_samples, variables=None):
    """Return KL divergence between states in engine_1 from engine_0."""
    # Use all variables in the population by default.
    if variables is None:
        variables = engine_0.states[0].outputs

    # Generate samples from the states in engine_0.
    # samples_0[i][k] contains sample k from engine_0.states[i].
    samples_0 = engine_0.simulate(rowid=None, query=variables, N=num_samples)

    # Compute logpdfs of samples_0 by states from engine_0.
    # logpdfs_0[i][k] contains logpdf of sample k from engine_0.states[i]
    # assessed by engine_0.states[i].
    rowids = [[-1]*num_samples] * len(samples_0)
    logpdfs_0 = logpdf_bulk_heterogeneous(engine_0, rowids, samples_0)

    # Compute logpdfs of samples_0 by states from engine_1.
    unraveled_samples_0 = np.ravel(samples_0)
    rowids = [-1] * len(unraveled_samples_0)

    # unraveled_samples_1[j][i*num_samples + k] contains logpdf of sample k
    # from engine_0.states[i] assessed by engine.states[j].
    unraveled_logpdfs_1 = engine_1.logpdf_bulk(rowids, unraveled_samples_0)

    # unraveled_logpdfs_1[j][i][k] contains logpdf of sample k from
    # engine.states[i] assessed by engine_1.states[j].
    logpdfs_1 = np.reshape(
        unraveled_logpdfs_1,
        (engine_1.num_states(), engine_0.num_states(), num_samples)
    )

    # The vectorized computation below has pairwise[i][j] as
    # KL(engines.states[i] || engine.states[j]), and is equivalent to:
    # pairwise_kl = np.array([
    #     [np.mean(logpdfs_0[i] - logpdfs_1[j][i])
    #         for j in xrange(engine_1.num_states())
    #     ] for i in xrange(engine_0.num_states())
    # ])
    pairwise_kl = np.transpose(
        np.mean(np.subtract(logpdfs_0, logpdfs_1), axis=-1))

    # Run sanity check.
    assert pairwise_kl.shape == (engine_0.num_states(), engine_1.num_states())

    return pairwise_kl


def compute_predictive_kl(engine_0, engine_1, num_samples, variables=None):
    """Return KL divergence between states in engine_1 from engine_0."""
    # Use all variables in the population by default.
    if variables is None:
        variables = engine_0.states[0].outputs

    # Generate samples from the states in engine_0.
    # samples_0[i][k] contains sample k from engine_0.states[i].
    samples_0 = engine_0.simulate(rowid=None, query=variables, N=num_samples)

    # Resample (no replacement), to simulate the posterior predictive.
    resamples = engine_0.rng.choice(
        np.ravel(samples_0), size=num_samples, replace=False)

    # Compute logpdfs of resamples using engine_0.
    rowids = [-1] * num_samples
    logpdfs_0 = np.array(engine_0.logpdf_bulk(rowids, resamples))
    logpdfs_1 = np.array(engine_1.logpdf_bulk(rowids, resamples))

    # Compute predictive logpdf of resamples under engine_0.
    predictive_logpdfs_0 = [
        engine_0._likelihood_weighted_integrate(logpdfs_0[:,i], -1)
        for i in xrange(num_samples)
    ]
    # Compute predictive logpdf of resamples under engine_1.
    predictive_logpdfs_1 = [
        engine_1._likelihood_weighted_integrate(logpdfs_1[:,i], -1)
        for i in xrange(num_samples)
    ]

    # Run sanity check.
    assert len(predictive_logpdfs_0) == num_samples
    assert len(predictive_logpdfs_1) == num_samples

    kl = np.mean(np.subtract(predictive_logpdfs_0, predictive_logpdfs_1))

    return kl

def logpdf_bulk_heterogeneous(engine, rowids, queries, evidences=None,
        multiprocess=True):
    """Compute the logpdf in bulk, with different queries per state."""
    evidences = evidences or [None]*len(queries)
    mapper = parallel_map if multiprocess else map
    args = [('logpdf_bulk', engine.states[i],
            (rowids[i], queries[i], evidences[i]))
            for i in xrange(engine.num_states())]
    logpdfs = mapper(_engine_evaluate, args)
    return logpdfs
