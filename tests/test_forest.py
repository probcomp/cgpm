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

import pytest

from math import log

import numpy as np

from scipy.misc import logsumexp

from gpmcc.mixtures.dim import Dim
from gpmcc.dists.forest import RandomForest
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


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
RF_DISTARGS = {'k': distargs[0]['k'], 'cctypes': cctypes[1:]}
RF_OUTPUTS = [0]
RF_INPUTS = range(1, len(cctypes))
NUM_CLASSES = 3


def test_incorporate():
    forest = RandomForest(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS,
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    # Incorporate first 20 rows.
    for rowid, row in enumerate(D[:20]):
        query = {0: row[0]}
        evidence = {i: row[i] for i in forest.inputs}
        forest.incorporate(rowid, query, evidence)
    # Unincorporating row 20 should raise.
    with pytest.raises(KeyError):
        forest.unincorporate(20)
    # Unincorporate all rows.
    for rowid in xrange(20):
        forest.unincorporate(rowid)
    # Unincorporating row 0 should raise.
    with pytest.raises(KeyError):
        forest.unincorporate(0)
    # Incorporating with wrong covariate dimensions should raise.
    with pytest.raises(TypeError):
        query = {0: D[0,0]}
        evidence = {i: v for (i, v) in enumerate(D[0])}
        forest.incorporate(0, query, evidence)
    # Incorporate some more rows.
    for rowid, row in enumerate(D[:10]):
        query = {0: row[0]}
        evidence = {i: row[i] for i in forest.inputs}
        forest.incorporate(rowid, query, evidence)


def test_logpdf_uniform():
    """No observations implies uniform."""
    forest = RandomForest(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS,
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    forest.transition_params()
    for x in xrange(NUM_CLASSES):
        query = {0: x}
        evidence = {i: D[0,i] for i in forest.inputs}
        assert np.allclose(
            forest.logpdf(-1, query, evidence), -log(NUM_CLASSES))


def test_logpdf_normalized():
    def train_on(c):
        D_sub = [(i, row) for (i, row) in enumerate(D) if row[0] in c]
        forest = RandomForest(
            outputs=RF_OUTPUTS, inputs=RF_INPUTS,
            distargs=RF_DISTARGS, rng=gu.gen_rng(0))
        for rowid, row in D_sub:
            query = {0: row[0]}
            evidence = {i: row[i] for i in forest.inputs}
            forest.incorporate(rowid, query, evidence)
        forest.transition_params()
        return forest

    def test_on(forest, c):
        D_sub = [(i, row) for (i, row) in enumerate(D) if row[0] not in c]
        for rowid, row in D_sub:
            evidence = {i: row[i] for i in forest.inputs}
            queries =[{0: x} for x in xrange(NUM_CLASSES)]
            lps = [forest.logpdf(rowid, q, evidence) for q in queries]
            assert np.allclose(logsumexp(lps), 0)

    forest = train_on([])
    test_on(forest, [])

    forest = train_on([2])
    test_on(forest, [2])

    forest = train_on([0,1])
    test_on(forest, [0,1])


def test_logpdf_score():
    forest = RandomForest(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS,
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    for rowid, row in enumerate(D[:25]):
        query = {0: row[0]}
        evidence = {i: row[i] for i in forest.inputs}
        forest.incorporate(rowid, query, evidence)
    forest.transition_params()
    assert forest.logpdf_score() < 0


def test_transition_hypers():
    forest = Dim(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS, cctype='random_forest',
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    forest.transition_hyper_grids(D[:,0])
    # Create two clusters.
    Zr = np.zeros(len(D), dtype=int)
    Zr[len(D)/2:] = 1
    for rowid, row in enumerate(D[:25]):
        query = {0: row[0]}
        evidence = gu.merge_dicts(
            {i: row[i] for i in forest.inputs}, {-1: Zr[rowid]})
        forest.incorporate(rowid, query, evidence)
    # Transitions.
    forest.transition_params()
    forest.transition_hypers()


def test_simulate():
    forest = Dim(
        outputs=RF_OUTPUTS, inputs=RF_INPUTS, cctype='random_forest',
        distargs=RF_DISTARGS, rng=gu.gen_rng(0))
    forest.transition_hyper_grids(D[:,0])
    # Incorporate data into 1 cluster.
    for rowid, row in enumerate(D[:40]):
        query = {0: row[0]}
        evidence = gu.merge_dicts({i:row[i] for i in forest.inputs}, {-1:0})
        forest.incorporate(rowid, query, evidence)
    # Transitions.
    forest.transition_params()
    for _ in xrange(2):
        forest.transition_hypers()
    correct, total = 0, 0.
    for row in D[40:]:
        evidence = gu.merge_dicts({i:row[i] for i in forest.inputs}, {-1:0})
        s = [forest.simulate(-1, [0], evidence) for _ in xrange(10)]
        s = np.argmax(np.bincount(s))
        correct += (s==row[0])
        total += 1.
    # Classification should be better than random.
    assert correct/total > 1./NUM_CLASSES
