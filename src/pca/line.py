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

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from cgpm.pca.ppca import PPCA
from cgpm.utils import general as gu
from cgpm.utils import config as cu

rng = gu.gen_rng(12)

# Number of datapoints.
N = 100

# First principal component vector.
W = np.asarray([
    [3.,],
    [20.],
    [-4.],])

# First principal component scores.
X = rng.normal(size=(1,N))

# Mean vector.
m = np.asarray([
    [1],
    [2],
    [-3],])
M = np.repeat(m, repeats=N, axis=1)

# Noise vector
sigma = 1.9
e = rng.normal(scale=sigma, size=(3,N))

# Observations.
Y = np.dot(W,X) + M + e

assert Y.shape == (3, N)

fig, ax = plt.subplots()

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)
fa.fit(Y.T)

def transform(fa, x):
    dot, inv = np.dot, np.linalg.inv
    L, D = fa.components_.shape
    mu = fa.mean_
    assert mu.shape == (D,)
    Phi = np.diag(fa.noise_variance_)
    assert Phi.shape == (D, D)
    W = fa.components_.T
    assert W.shape == (D, L)
    I = np.eye(L)
    S = inv((I + dot(W.T, dot(inv(Phi), W))))
    m = dot(S, dot(W.T, dot(inv(Phi), (x-mu))))
    return m

for x in Y.T:
    print x
    result_a = fa.transform([x])
    result_b = transform(fa, x)
    assert np.allclose(result_a, result_b)