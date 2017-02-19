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
import pytest

import numpy as np

from cgpm.primitives.crp import Crp
from cgpm.utils import general as gu


def simulate_crp_gpm(N, alpha, rng):
    crp = Crp(outputs=[0], inputs=None, hypers={'alpha':alpha}, rng=rng)
    for i in xrange(N):
        s = crp.simulate(i, [0], None)
        crp.incorporate(i, s, None)
    return crp


def assert_crp_equality(alpha, Nk, crp):
    N = sum(Nk)
    Z = list(itertools.chain.from_iterable(
        [i]*n for i, n in enumerate(Nk)))
    P = crp.data.values()
    assert len(Z) == len(P) == N
    probe_values = set(P).union({max(P)+1})
    assert Nk == crp.counts.values()
    # Table predictive probabilities.
    assert np.allclose(
        gu.logp_crp_fresh(N, Nk, alpha),
        [crp.logpdf(-1, {0:v}, None) for v in probe_values])
    # Data probability.
    assert np.allclose(
        gu.logp_crp(N, Nk, alpha),
        crp.logpdf_score())
    # Gibbs transition probabilities.
    Z = crp.data.values()
    for i, rowid in enumerate(crp.data):
        assert np.allclose(
            gu.logp_crp_gibbs(Nk, Z, i, alpha, 1),
            crp.gibbs_logps(rowid))


N = [2**i for i in xrange(8)]
alpha = gu.log_linspace(.001, 100, 10)
seed = [5]


@pytest.mark.parametrize('N, alpha, seed', itertools.product(N, alpha, seed))
def test_crp_simple(N, alpha, seed):
    # Obtain the partitions.
    A = gu.simulate_crp(N, alpha, rng=gu.gen_rng(seed))
    Nk = list(np.bincount(A))

    crp = simulate_crp_gpm(N, alpha, rng=gu.gen_rng(seed))

    assert A == crp.data.values()
    assert_crp_equality(alpha, Nk, crp)


@pytest.mark.parametrize('N, alpha, seed', itertools.product(N, alpha, seed))
def test_crp_decrement(N, alpha, seed):
    A = gu.simulate_crp(N, alpha, rng=gu.gen_rng(seed))
    Nk = list(np.bincount(A))
    # Decrement all counts by 1.
    Nk = [n-1 if n > 1 else n for n in Nk]

    # Decrement rowids.
    crp = simulate_crp_gpm(N, alpha, rng=gu.gen_rng(seed))
    targets = [c for c in crp.counts if crp.counts[c] > 1]
    seen = set([])
    for r, c in crp.data.items():
        if c in targets and c not in seen:
            seen.add(c)
            crp.unincorporate(r)
        if seen == len(targets):
            break

    assert_crp_equality(alpha, Nk, crp)


@pytest.mark.parametrize('N, alpha, seed', itertools.product(N, alpha, seed))
def test_crp_increment(N, alpha, seed):
    A = gu.simulate_crp(N, alpha, rng=gu.gen_rng(seed))
    Nk = list(np.bincount(A))
    # Add 3 new classes.
    Nk.extend([2, 3, 1])

    crp = simulate_crp_gpm(N, alpha, rng=gu.gen_rng(seed))
    # Increment rowids.
    rowid = max(crp.data)
    clust = max(crp.data.values())
    crp.incorporate(rowid+1, {0:clust+1}, None)
    crp.incorporate(rowid+2, {0:clust+1}, None)
    crp.incorporate(rowid+3, {0:clust+2}, None)
    crp.incorporate(rowid+4, {0:clust+2}, None)
    crp.incorporate(rowid+5, {0:clust+2}, None)
    crp.incorporate(rowid+6, {0:clust+3}, None)

    assert_crp_equality(alpha, Nk, crp)

def test_gibbs_tables_logps():
    crp = Crp(
        outputs=[0], inputs=None, hypers={'alpha': 1.5}, rng=gu.gen_rng(1))

    assignments = [
        (0, {0: 0}),
        (1, {0: 0}),
        (2, {0: 2}),
        (3, {0: 2}),
        (4, {0: 2}),
        (5, {0: 6}),
    ]

    for rowid, table in assignments:
        crp.incorporate(rowid, table, None)

    # customer 0
    K01 = crp.gibbs_tables(0, m=1)
    assert K01 == [0, 2, 6, 7]
    P01 = crp.gibbs_logps(0, m=1)
    assert np.allclose(np.exp(P01), [1, 3, 1, 1.5])

    K02 = crp.gibbs_tables(0, m=2)
    assert K02 == [0, 2, 6, 7, 8]
    P02 = crp.gibbs_logps(0, m=2)
    assert np.allclose(np.exp(P02), [1, 3, 1, 1.5/2, 1.5/2])

    K03 = crp.gibbs_tables(0, m=3)
    assert K03 == [0, 2, 6, 7, 8, 9]
    P03 = crp.gibbs_logps(0, m=3)
    assert np.allclose(np.exp(P03), [1, 3, 1, 1.5/3, 1.5/3, 1.5/3])

    K21 = crp.gibbs_tables(2, m=1)
    assert K21 == [0, 2, 6, 7]
    P21 = crp.gibbs_logps(2, m=1)
    assert np.allclose(np.exp(P21), [2, 2, 1, 1.5])

    K22 = crp.gibbs_tables(2, m=2)
    assert K22 == [0, 2, 6, 7, 8]
    P22 = crp.gibbs_logps(2, m=2)
    assert np.allclose(np.exp(P22), [2, 2, 1, 1.5/2, 1.5/2])

    K23 = crp.gibbs_tables(2, m=3)
    P23 = crp.gibbs_logps(2, m=3)
    assert K23 == [0, 2, 6, 7, 8, 9]
    assert np.allclose(np.exp(P23), [2, 2, 1, 1.5/3, 1.5/3, 1.5/3])

    K51 = crp.gibbs_tables(5, m=1)
    assert K51 == [0, 2, 6]
    P51 = crp.gibbs_logps(5, m=1)
    assert np.allclose(np.exp(P51), [2, 3, 1.5])

    K52 = crp.gibbs_tables(5, m=2)
    assert K52 == [0, 2, 6, 7]
    P52 = crp.gibbs_logps(5, m=2)
    assert np.allclose(np.exp(P52), [2, 3, 1.5/2, 1.5/2])

    K53 = crp.gibbs_tables(5, m=3)
    P53 = crp.gibbs_logps(5, m=3)
    assert K53 == [0, 2, 6, 7, 8]
    assert np.allclose(np.exp(P53), [2, 3, 1.5/3, 1.5/3, 1.5/3])


def test_crp_logpdf_score():
    """Ensure that logpdf_marginal agrees with sequence of predictives."""
    crp = Crp(
        outputs=[0], inputs=None, hypers={'alpha': 1.5}, rng=gu.gen_rng(1))

    assignments = [
        (0, {0: 0}),
        (1, {0: 0}),
        (2, {0: 2}),
        (3, {0: 2}),
        (4, {0: 2}),
        (5, {0: 6}),
    ]

    logpdf_predictive = []
    logpdf_marginal = []

    # Incorporate rows record predictive/marginal logpdfs.
    for (rowid, query) in assignments:
        logpdf_predictive.append(crp.logpdf(rowid, query))
        crp.incorporate(rowid, query)
        logpdf_marginal.append(crp.logpdf_score())

    # Joint probability = sum of predictives = marginal likelihood.
    assert np.allclose(sum(logpdf_predictive), logpdf_marginal[-1])

    # First differences of logpdf marginal should be predictives.
    assert np.allclose(np.diff(logpdf_marginal), logpdf_predictive[1:])

    # Ensure that marginal depends on the actual data.
    crp.unincorporate(rowid=5)
    crp.incorporate(5, {0:0})
    assert not np.allclose(crp.logpdf_score(), logpdf_marginal[-1])
