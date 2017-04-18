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


import itertools

import numpy as np
import pytest

from cgpm.crosscat.engine import Engine

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu

from cgpm.crosscat import kl


def retrieve_engine(num_states=16, n_rows=10, n_iters=100):
    """Retrieve a generic engine."""
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'normal',
        'categorical(k=4)'
    ])
    (T, _Zv, _Zc) = tu.gen_data_table(
        n_rows=n_rows,
        view_weights=[1],
        cluster_weights=[[.25, .25, .5]],
        cctypes=cctypes,
        distargs=distargs,
        separation=[.95]*len(cctypes),
        rng=gu.gen_rng(10)
    )
    engine = Engine(
        T.T, cctypes=cctypes, distargs=distargs, rng=gu.gen_rng(312),
        num_states=num_states)
    engine.transition_lovecat(N=n_iters, progress=False)
    return engine


def test_logpdf_bulk_heterogeneous():
    """Test the ad-hoc implementation of logpdf_bulk_heterogeneous."""
    engine = retrieve_engine(16)

    variables = [0,1]
    num_samples = 5
    samples = engine.simulate(rowid=None, query=variables, N=num_samples)

    queries = samples
    rowids = [[-1]*num_samples for _i  in xrange(engine.num_states())]
    logpdfs_bulk = kl.logpdf_bulk_heterogeneous(engine, rowids, queries)

    for (i, (sample, state)) in enumerate(zip(queries, engine.states)):
        logpdfs = state.logpdf_bulk([-1]*len(sample), sample)
        assert np.allclose(logpdfs_bulk[i], logpdfs)

    # Remove a query point.
    queries[0].pop()
    queries[-1].pop()

    # Wrong dimension of rowids.
    with pytest.raises(AssertionError):
        logpdfs_bulk = kl.logpdf_bulk_heterogeneous(engine, rowids, queries)

    # Use the correct number of rowids.
    rowids[0].pop()
    rowids[-1].pop()
    logpdfs_bulk = kl.logpdf_bulk_heterogeneous(engine, rowids, queries)
    assert len(logpdfs_bulk[0]) == num_samples - 1
    assert len(logpdfs_bulk[-1]) == num_samples - 1
    assert all(len(lp) == num_samples for lp in logpdfs_bulk[1:-1])

    for (i, (sample, state)) in enumerate(zip(queries, engine.states)):
        logpdfs = state.logpdf_bulk([-1]*len(sample), sample)
        assert np.allclose(logpdfs_bulk[i], logpdfs)


def compute_kl(n_rows, n_iters):
    engine = retrieve_engine(64, n_rows, n_iters)
    engine_0 = engine
    engine_1 = engine
    num_samples = 200
    pairwise_kl = kl.compute_pairwise_kl(engine_0, engine_1, num_samples)
    np.savetxt(
        'resources/kl_rows=%03d_iters=%03d' % (n_rows, n_iters),
        pairwise_kl,
        delimiter=',',
    )

n_rows = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_iters = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

for nr, ni in itertools.product(n_rows, n_iters):
    print 'Computing: n_rows=%03d, iters=%03d' % (nr, ni)
    compute_kl(nr, ni)
