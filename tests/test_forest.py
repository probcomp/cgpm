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
import pytest

from math import log

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

from cgpm.mixtures.dim import Dim
from cgpm.regressions.forest import RandomForest
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu

from stochastic import stochastic


cctypes, distargs = cu.parse_distargs([
    'categorical(k=3)',
    'normal',
    'poisson',
    'bernoulli',
    'lognormal',
    'exponential',
    'geometric',
    'vonmises'])

T, Zv, Zc = tu.gen_data_table(
    50, [1], [[.33, .33, .34]], cctypes, distargs,
    [.2]*len(cctypes), rng=gu.gen_rng(0))

D = T.T
RF_DISTARGS = {'inputs': {'stattypes': cctypes[1:]}, 'k': distargs[0]['k']}
RF_OUTPUTS = [0]
RF_INPUTS = range(1, len(cctypes))
NUM_CLASSES = 3


def test_incorporate():
    forest = RandomForest(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS,
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    # Incorporate first 20 rows.
    for rowid, row in enumerate(D[:20]):
        observation = {0: row[0]}
        inputs = {i: row[i] for i in forest.inputs}
        forest.incorporate(rowid, observation, inputs)
    # Unincorporating row 20 should raise.
    with pytest.raises(ValueError):
        forest.unincorporate(20)
    # Unincorporate all rows.
    for rowid in xrange(20):
        forest.unincorporate(rowid)
    # Unincorporating row 0 should raise.
    with pytest.raises(ValueError):
        forest.unincorporate(0)
    # Incorporating with wrong covariate dimensions should raise.
    with pytest.raises(ValueError):
        observation = {0: D[0,0]}
        inputs = {i: v for (i, v) in enumerate(D[0])}
        forest.incorporate(0, observation, inputs)
    # Incorporating with wrong output categorical value should raise.
    with pytest.raises(IndexError):
        observation = {0: 100}
        inputs = {i: D[0,i] for i in forest.inputs}
        forest.incorporate(0, observation, inputs)
    # Incorporating with nan inputs value should raise.
    with pytest.raises(ValueError):
        observation = {0: 100}
        inputs = {i: D[0,i] for i in forest.inputs}
        inputs[inputs.keys()[0]] = np.nan
        forest.incorporate(0, observation, inputs)
    # Incorporate some more rows.
    for rowid, row in enumerate(D[:10]):
        observation = {0: row[0]}
        inputs = {i: row[i] for i in forest.inputs}
        forest.incorporate(rowid, observation, inputs)


def test_logpdf_uniform():
    """No observations implies uniform."""
    forest = RandomForest(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS,
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    forest.transition_params()
    for x in xrange(NUM_CLASSES):
        targets = {0: x}
        inputs = {i: D[0,i] for i in forest.inputs}
        assert np.allclose(
            forest.logpdf(None, targets, None, inputs),
            -log(NUM_CLASSES),
        )


def test_logpdf_normalized():
    def train_one(c):
        D_sub = [(i, row) for (i, row) in enumerate(D) if row[0] in c]
        forest = RandomForest(
            outputs=RF_OUTPUTS, inputs=RF_INPUTS,
            distargs=RF_DISTARGS, rng=gu.gen_rng(0))
        for rowid, row in D_sub:
            observation = {0: row[0]}
            inputs = {i: row[i] for i in forest.inputs}
            forest.incorporate(rowid, observation, inputs)
        forest.transition_params()
        return forest

    def test_one(forest, c):
        D_sub = [(i, row) for (i, row) in enumerate(D) if row[0] not in c]
        for rowid, row in D_sub:
            inputs = {i: row[i] for i in forest.inputs}
            targets =[{0: x} for x in xrange(NUM_CLASSES)]
            lps = [forest.logpdf(rowid, q, None, inputs) for q in targets]
            assert np.allclose(gu.logsumexp(lps), 0)

    forest = train_one([])
    test_one(forest, [])

    forest = train_one([2])
    test_one(forest, [2])

    forest = train_one([0,1])
    test_one(forest, [0,1])


def test_logpdf_score():
    forest = RandomForest(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS,
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    for rowid, row in enumerate(D[:25]):
        observation = {0: row[0]}
        inputs = {i: row[i] for i in forest.inputs}
        forest.incorporate(rowid, observation, inputs)
    forest.transition_params()
    forest.transition_params()

    logscore = forest.logpdf_score()
    assert logscore < 0

    # Use a deserialized version for simulating.
    metadata = forest.to_metadata()
    builder = getattr(
        importlib.import_module(metadata['factory'][0]),
        metadata['factory'][1])
    forest2 = builder.from_metadata(metadata, rng=gu.gen_rng(1))

    assert forest2.alpha == forest.alpha
    assert np.allclose(forest2.counts, forest.counts)
    assert np.allclose(forest2.logpdf_score(), logscore)


def test_transition_hypers():
    forest = Dim(
        outputs=RF_OUTPUTS, inputs=[-1]+RF_INPUTS, cctype='random_forest',
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    forest.transition_hyper_grids(D[:,0])

    # Create two clusters.
    Zr = np.zeros(len(D), dtype=int)
    Zr[len(D)/2:] = 1
    for rowid, row in enumerate(D[:25]):
        observation = {0: row[0]}
        inputs = gu.merged(
            {i: row[i] for i in forest.inputs}, {-1: Zr[rowid]})
        forest.incorporate(rowid, observation, inputs)


@stochastic(max_runs=3, min_passes=1)
def test_simulate(seed):
    rng = gu.gen_rng(bytearray(seed))

    iris = load_iris()
    indices = rng.uniform(0, 1, size=len(iris.data)) <= .75

    Y_train = iris.data[indices]
    X_train = iris.target[indices]

    Y_test = iris.data[~indices]
    X_test = iris.target[~indices]

    forest = Dim(
        outputs=[5], inputs=[-1]+range(4), cctype='random_forest',
        distargs={
            'inputs': {'stattypes': ['normal']*4},
            'k': len(iris.target_names)},
        rng=rng)

    forest.transition_hyper_grids(X_test)

    # Incorporate data into 1 cluster.
    for rowid, (x, y) in enumerate(zip(X_train, Y_train)):
        observation = {5: x}
        inputs = gu.merged(
            {-1: 0},
            {i: t for (i,t) in zip(range(4), y)})
        forest.incorporate(rowid, observation, inputs)

    # Transitions.
    for _i in xrange(2):
        forest.transition_hypers()
        forest.transition_params()

    correct, total = 0, 0.
    for rowid, (x, y) in enumerate(zip(X_test, Y_test)):
        inputs = gu.merged(
            {-1: 0},
            {i: t for (i,t) in zip(range(4), y)})
        samples = forest.simulate(None, [5], None, inputs, 10)
        prediction = np.argmax(np.bincount([s[5] for s in samples]))
        correct += (prediction==x)
        total += 1.

    # Classification should be better than random.
    assert correct/total > 1./ forest.distargs['k']
