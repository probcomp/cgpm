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

"""This test suite ensures that results returned from Engine.logpdf,
Engine.logpdf_bulk, Engine.simulate, and Engine.simulate_bulk are of the
correct dimensions, based on the number of states and type of targets.

Every results from these queries should be a list of length
Engine.num_states(). The elements of the returned list differ based on the
method, where we use

    - logpdf[s] = logpdf of targets from state s.

    - simulate[s][i] = sample i (i=1..N) of targets from state s.

    - logpdf_bulk[s][j] = logpdf of targets[j] from state s.

    - simulate_bulk[s][j][i] = sample (i=1..Ns[j]) of targets[j] from state s.

This test suite is slow because many simulate/logpdf queries are invoked.
"""

from builtins import range
import numpy as np
import pytest

from cgpm.crosscat.engine import Engine
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


@pytest.fixture(scope='module')
def engine():
    # Set up the data generation
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'lognormal',
        'beta',
        'vonmises'
    ])
    T, Zv, Zc = tu.gen_data_table(
        20, [1], [[.25, .25, .5]], cctypes, distargs,
        [.95]*len(cctypes), rng=gu.gen_rng(0))
    T = T.T
    # Make some nan cells for constraints.
    T[5,0] = T[5,1] = T[5,2] = T[5,3] = np.nan
    T[8,4] = np.nan
    e = Engine(
        T,
        cctypes=cctypes,
        distargs=distargs,
        num_states=6,
        rng=gu.gen_rng(0)
    )
    e.transition(N=2)
    return e.to_metadata()


def test_logpdf__ci_(engine):
    engine = Engine.from_metadata(engine)

    def test_correct_dimensions(rowid, targets, constraints, statenos):
        # logpdfs should be a list of floats.
        logpdfs = engine.logpdf(
            rowid, targets, constraints=constraints, statenos=statenos)
        assert len(logpdfs) == (
            engine.num_states() if statenos is None else len(statenos))
        for state_logpdfs in logpdfs:
            # Each element in logpdfs should be a single float.
            assert isinstance(state_logpdfs, float)
        lp = engine._likelihood_weighted_integrate(
            logpdfs, rowid, constraints=constraints, statenos=statenos)
        assert isinstance(lp, float)

    for statenos in (None, [0, 2, 4]):
        test_correct_dimensions(-1, {0:1}, {2:1, 3:.5}, statenos)
        test_correct_dimensions(-1, {2:0, 5:3}, {0:4, 1:5}, statenos)
        test_correct_dimensions(5, {0:0, 2:0}, {3:3}, statenos)


def test_simulate__ci_(engine):
    engine = Engine.from_metadata(engine)

    def test_correct_dimensions(rowid, targets, constraints, N, statenos):
        samples = engine.simulate(
            rowid, targets, constraints=constraints, N=N, statenos=statenos)
        assert len(samples) == (
            engine.num_states() if statenos is None else len(statenos))
        for states_samples in samples:
            # Each element of samples should be a list of N samples.
            assert len(states_samples) == N
            for s in states_samples:
                # Each raw sample should be len(Q) dimensional.
                assert set(s.keys()) == set(targets)
                assert len(s) == len(targets)
        s = engine._likelihood_weighted_resample(
            samples, rowid, constraints=constraints, statenos=statenos)
        assert len(s) == N

    targets1, constraints1 = [0], {2:0, 3:6}
    targets2, constraints2 = [1,2,5], {0:3, 3:.8}

    for statenos in (None, (1,3)):
        test_correct_dimensions(-1, targets1, constraints1, 1, statenos)
        test_correct_dimensions(-1, targets1, constraints1, 8, statenos)
        test_correct_dimensions(-1, targets2, constraints2, 1, statenos)
        test_correct_dimensions(-1, targets2, constraints2, 8, statenos)

        targets3, constraints3 = [0,1,2], {3:1}
        test_correct_dimensions(5, targets3, constraints3, 1, statenos)
        test_correct_dimensions(5, targets3, constraints3, 8, statenos)


def test_logpdf_bulk__ci_(engine):
    engine = Engine.from_metadata(engine)
    rowid1, targets1, constraints1 = 5, {0:0}, {2:1, 3:.5}
    rowid2, targets2, constraints2 = -1, {1:0, 4:.8}, {5:.5}
    # Bulk.
    rowids = [rowid1, rowid2]
    targets_list = [targets1, targets2]
    constraints_list = [constraints1, constraints2]

    def test_correct_dimensions(statenos):
        # Invoke
        logpdfs = engine.logpdf_bulk(rowids, targets_list,
            constraints_list=constraints_list, statenos=statenos)
        assert len(logpdfs) == \
            engine.num_states() if statenos is None else len(statenos)
        for state_logpdfs in logpdfs:
            # state_logpdfs should be a list of floats, one float per targets.
            assert len(state_logpdfs) == len(rowids)
            for l in state_logpdfs:
                    assert isinstance(l, float)

    test_correct_dimensions(statenos=None)
    test_correct_dimensions(statenos=[0, 1, 4, 5])


def test_simulate_bulk__ci_(engine):
    engine = Engine.from_metadata(engine)
    rowid1, targets1, constraints1, N1, = -1, [0,2,4,5], {3:1}, 7
    rowid2, targets2, constraints2, N2 = 5, [1,3], {2:1}, 3
    rowid3, targets3, constraints3, N3 = 8, [0], {4:.8}, 3
    # Bulk.
    rowids = [rowid1, rowid2, rowid3]
    targets_list = [targets1, targets2, targets3]
    constraints_list = [constraints1, constraints2, constraints3]
    Ns = [N1, N2, N3]

    def test_correct_dimensions(statenos):
        # Invoke
        samples = engine.simulate_bulk(
            rowids, targets_list, constraints_list=constraints_list,
            Ns=Ns, statenos=statenos)
        assert len(samples) == (
            engine.num_states() if statenos is None else len(statenos))
        for states_samples in samples:
            assert len(states_samples) == len(rowids)
            for i, sample in enumerate(states_samples):
                assert len(sample) == Ns[i]
                for s in sample:
                    assert set(s.keys()) == set(targets_list[i])
                    assert len(s) == len(targets_list[i])

    test_correct_dimensions(None)
    test_correct_dimensions([4])


def test_dependence_probability__ci_(engine):
    engine = Engine.from_metadata(engine)

    results = engine.dependence_probability(0, 2, statenos=None)
    assert len(results) == engine.num_states()

    results = engine.dependence_probability(0, 2, statenos=[1,4])
    assert len(results) == 2

def test_row_similarity__ci_(engine):
    engine = Engine.from_metadata(engine)

    results = engine.row_similarity(0, 2, statenos=None)
    assert len(results) == engine.num_states()

    results = engine.row_similarity(0, 2, statenos=[1,4,5])
    assert len(results) == 3

def test_relevance_probability__ci_(engine):
    engine = Engine.from_metadata(engine)

    results = engine.relevance_probability(0, [2, 14], 0, statenos=None)
    assert len(results) == engine.num_states()

    results = engine.relevance_probability(
        0, [2, 14], 0, statenos=list(range(engine.num_states())))
    assert len(results) == engine.num_states()
