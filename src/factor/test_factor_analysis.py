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

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.decomposition

from cgpm.utils import general as gu
from cgpm.factor.factor import FactorAnalysis


def scatter_classes(x, classes, ax=None):
    """Scatter the data points coloring by the classes."""
    if ax is None:
        _fig, ax = plt.subplots()
    ax = plt.gca() if ax is None else ax
    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.Normalize(
        vmin=np.min(classes), vmax=np.max(classes))
    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)
    ax.scatter(x[:,0], x[:,1], color=colors)
    return ax

def fillna(X, p, rng):
    """Population proportion p of entries in X with nan values."""
    X = np.copy(X)
    a, b = X.shape
    n_entries = a*b
    n_missing = int(a*b*p)
    i_missing_flat = rng.choice(range(n_entries), size=n_missing, replace=False)
    i_missing_cell = np.unravel_index(i_missing_flat, (a,b))
    for i, j in zip(*i_missing_cell):
        X[i,j] = np.nan
    return X


def test_invalid_initialize():
    # No inputs.
    with pytest.raises(ValueError):
        FactorAnalysis([1,2,6], [0], distargs={'L':1})
    # Missing L
    with pytest.raises(ValueError):
        FactorAnalysis([1,2,6], None, distargs=None)
    # Wrong dimensionality: no observables.
    with pytest.raises(ValueError):
        FactorAnalysis([1,2], None, distargs={'L':2})
    # Wrong dimensionality: latent space too big.
    with pytest.raises(ValueError):
        FactorAnalysis([1,2,3], None, distargs={'L':2})
    # Wrong dimensionality: latent space too small.
    with pytest.raises(ValueError):
        FactorAnalysis([1,2,3], None, distargs={'L':0})
    # Wrong dimensionality: not enough outputs.
    with pytest.raises(ValueError):
        FactorAnalysis([2], None, distargs={'L':1})
    # Duplicate outputs.
    with pytest.raises(ValueError):
        FactorAnalysis([2,2], None, distargs={'L':1})


def test_valid_initialize():
    # One latent dimension.
    fa = FactorAnalysis([4,2], None, distargs={'L':1})
    assert fa.D == 1
    assert fa.L == 1

    # Four latent dimensions.
    fa = FactorAnalysis(range(12), None, distargs={'L':4})
    assert fa.D == 8
    assert fa.L == 4

    # Latent dimension equal to observable dimensions.
    fa = FactorAnalysis([4,2,1,0,6,7], None, distargs={'L':3})
    assert fa.D == 3
    assert fa.L == 3


def test_incorporate():
    fa = FactorAnalysis([4,5,9,2], None, distargs={'L':1})
    # Cannot incorporate a latent variable.
    with pytest.raises(ValueError):
        fa.incorporate(0, {4:1, 5:1, 9:1, 0:0})
    # Cannot incorporate with evidence.
    with pytest.raises(ValueError):
        fa.incorporate(0, {4:1, 5:1, 9:1}, {0:0})
    # Need a query variable.
    with pytest.raises(ValueError):
        fa.incorporate(0, {})
    # Unknown variable.
    with pytest.raises(ValueError):
        fa.incorporate(0, {1:0})
    # Incorporate a full row.
    fa.incorporate(0, {4:1, 5:1, 9:1})
    assert fa.data[0] == [1,1,1]
    # Incorporate rows with missing data.
    fa.incorporate(2, {5:1, 9:1})
    assert fa.data[2] == [np.nan,1,1]
    # And another one.
    fa.incorporate(4, {9:1})
    assert fa.data[4] == [np.nan,np.nan,1]
    # And another one.
    fa.incorporate(6, {4:-1})
    assert fa.data[6] == [-1,np.nan,np.nan]

    for rowid in [0, 2, 4, 6]:
        fa.unincorporate(rowid)
    assert fa.N == 0
    assert fa.data == {}

    with pytest.raises(ValueError):
        fa.unincorporate(23)


# def retrieve_iris():
#     rng = gu.gen_rng(12)
#     iris = sklearn.datasets.load_iris()

#     cgpm = FactorAnalysis([0,1,2], None, distargs={'L':1})
#     for i, row in enumerate(iris.data):
#         # Cannot incorporate
#         with pytest.raises(ValueError):
#             cgpm.incorporate(i, {0:row[0], 1: row[1], 2:1})
#         cgpm.incorporate(i, {0:row[0], 1: row[1]})


def test_logpdf_rigorous():
    # Direct factor anaysis
    rng = gu.gen_rng(12)
    iris = sklearn.datasets.load_iris()

    fact = FactorAnalysis([5,8,10,12,-1], None, distargs={'L':1}, rng=rng)
    for i, row in enumerate(iris.data):
        fact.incorporate(i, {q:v for q,v in zip(fact.outputs, row)})

    fact.transition()

    for row in iris.data:
        # TEST 1: Posterior mean of the latent variable.
        dot, inv = np.dot, np.linalg.inv
        L, D = fact.fa.components_.shape
        mu = fact.fa.mean_
        assert mu.shape == (D,)
        Phi = np.diag(fact.fa.noise_variance_)
        assert Phi.shape == (D, D)
        W = fact.fa.components_.T
        assert W.shape == (D, L)
        I = np.eye(L)
        # Compute using Murphy explicitly.
        S1 = inv((I + dot(W.T, dot(inv(Phi), W))))
        m1 = dot(S1, dot(W.T, dot(inv(Phi), (row-mu))))
        # Compute using the Schur complement explicitly.
        S2 = I - dot(dot(W.T, inv(dot(W,W.T) + Phi)), W)
        m2 = dot(dot(W.T, inv(dot(W, W.T)+Phi)), (row-mu))
        # Compute the mean using the factor analyzer.
        m3 = fact.fa.transform([row])
        # Compute using the marginalize features of fact.fa.
        mG, covG = FactorAnalysis.mvn_condition(
            fact.mu, fact.cov,
            query=fact.reindex([-1]),
            evidence={
                fact.reindex([5])[0]: row[0],
                fact.reindex([8])[0]: row[1],
                fact.reindex([10])[0]: row[2],
                fact.reindex([12])[0]: row[3],
            })
        assert np.allclose(m1, m2)
        assert np.allclose(m2, m3)
        assert np.allclose(m3, mG)
        assert np.allclose(S1, S2)
        assert np.allclose(S2, covG)

        # TEST 2: Log density of observation.
