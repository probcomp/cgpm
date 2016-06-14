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

from gpmcc.exponentials.crp import Crp
from gpmcc.utils import general as gu


def simulate_crp_manual(N, alpha, rng):
    """Generates random N-length partition from the CRP with parameter alpha."""
    assert N > 0 and alpha > 0.
    alpha = float(alpha)
    partition = [0]*N
    Nk = [1]
    for i in xrange(1,N):
        K = len(Nk)
        ps = np.zeros(K+1)
        for k in xrange(K):
            ps[k] = float(Nk[k])
        ps[K] = alpha
        ps /= (float(i) - 1 + alpha)
        assignment = gu.pflip(ps, rng=rng)
        if assignment == K:
            Nk.append(1)
        elif assignment < K:
            Nk[assignment] += 1
        else:
            raise ValueError("Invalid assignment: %i, max=%i" % (assignment, K))
        partition[i] = assignment
    assert max(partition)+1 == len(Nk)
    assert len(partition)==N
    assert sum(Nk) == N
    K = len(Nk)
    return partition, Nk


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
    A, Nk = simulate_crp_manual(N, alpha, rng=gu.gen_rng(seed))
    crp = simulate_crp_gpm(N, alpha, rng=gu.gen_rng(seed))
    assert A == crp.data.values()
    assert_crp_equality(alpha, Nk, crp)


@pytest.mark.parametrize('N, alpha, seed', itertools.product(N, alpha, seed))
def test_crp_decrement(N, alpha, seed):
    A, Nk = simulate_crp_manual(N, alpha, rng=gu.gen_rng(seed))
    crp = simulate_crp_gpm(N, alpha, rng=gu.gen_rng(seed))

    # Decrement all counts by 1.
    Nk = [n-1 if n > 1 else n for n in Nk]
    # Decrement rowids.
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
    A, Nk = simulate_crp_manual(N, alpha, rng=gu.gen_rng(seed))
    crp = simulate_crp_gpm(N, alpha, rng=gu.gen_rng(seed))

    # Add 3 new classes.
    Nk.extend([2, 3, 1])

    # Decrement rowids.
    rowid = max(crp.data)
    clust = max(crp.data.values())
    crp.incorporate(rowid+1, {0:clust+1}, None)
    crp.incorporate(rowid+2, {0:clust+1}, None)
    crp.incorporate(rowid+3, {0:clust+2}, None)
    crp.incorporate(rowid+4, {0:clust+2}, None)
    crp.incorporate(rowid+5, {0:clust+2}, None)
    crp.incorporate(rowid+6, {0:clust+3}, None)

    assert_crp_equality(alpha, Nk, crp)
