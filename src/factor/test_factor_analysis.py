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
