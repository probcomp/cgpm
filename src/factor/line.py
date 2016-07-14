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

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal

from cgpm.utils import config as cu
from cgpm.utils import general as gu


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

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=1)
fa.fit(Y.T)

# Hard code posterior.
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
    # Compute using Murphy.
    S1 = inv((I + dot(W.T, dot(inv(Phi), W))))
    m1 = dot(S1, dot(W.T, dot(inv(Phi), (x-mu))))
    # Compute using the Schur complement.
    S2 = I - dot(dot(W.T, inv(dot(W,W.T) + Phi)), W)
    m2 = dot(dot(W.T, inv(dot(W, W.T)+Phi)), (x-mu))
    assert np.allclose(m1, m2)
    assert np.allclose(S1, S2)
    print m1, m2
    return m1

def logp(fa, x):
    dot, inv = np.dot, np.linalg.inv
    L, D = fa.components_.shape
    mu = fa.mean_
    assert mu.shape == (D,)
    Phi = np.diag(fa.noise_variance_)
    assert Phi.shape == (D, D)
    W = fa.components_.T
    assert W.shape == (D, L)
    return multivariate_normal.logpdf(x, mu, Phi + np.dot(W, W.T))

for x in Y.T:
    print x
    # Compute the latent variables z.
    z_a = fa.transform(np.asarray([x]))
    z_b = transform(fa, x)
    assert np.allclose(z_a, z_b)
    # Compute the probability p(x), marginalizing over the latent variables.
    lp_a = fa.score(np.asarray([x]))
    lp_b = logp(fa, x)
    assert np.allclose(lp_a, lp_b)


def mvn_marginalize(mu, cov, query, evidence):
    # Q = [0]
    # E = [1,3,4]
    # QE = Q+E
    mu = np.random.rand(5)
    cov = np.outer(mu, mu)
    # Query and evidence indices.
    Q = query
    E = evidence
    # QE = Q+E
    # muQ = mu[Q]
    # muE = mu[E]
    # muQE = mu[Q+E]
    covQ = cov[Q][:,Q]
    covE = cov[E][:,E]
    covJ = cov[Q][:,E]
    top = np.column_stack((covQ, covJ))
    bottom = np.column_stack((covJ.T, covE))
    covQE = np.row_stack((top, bottom))
    assert np.allclose(covQE, covQE.T)
    return mu[Q], mu[E], covQ, covE, covJ


def mvn_condition(mu, cov, query, evidence):
    # assert isinstance(query, list)
    # assert isinstance(evidence, dict)
    # assert len(mu) == cov.shape[0] == cov.shape[1]
    # assert len(query) + len(evidence) <= len(mu)
    Q, E = sorted(query), sorted(evidence.keys())
    Ev = np.asarray([evidence[e] for e in E])
    muQ, muE, covQ, covE, covJ = mvn_marginalize(mu, cov, Q, E)
    P = np.dot(covJ, np.linalg.inv(covE))
    muZ = muQ + np.dot(P, Ev - muE)
    covZ = covQ - np.dot(P, covJ.T)


# Number of datapoints.
N = 100
# First principal component vector.
W = np.asarray([
    [3.,],
    [20.]])
# First principal component scores.
X = rng.normal(size=(1,N))
# Mean vector.
m = np.asarray([
    [1],
    [2]])
M = np.repeat(m, repeats=N, axis=1)
# Noise vector
e = np.column_stack((
    rng.normal(scale=1.9, size=N), rng.normal(scale=3, size=N))).T
# Observations.
Y = np.dot(W,X) + M + e
assert Y.shape == (2, N)

# Plot the raw data.
fig, ax = plt.subplots()
ax.scatter(Y.T[:,0], Y.T[:,1], color='b')
ax.grid()

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=1)
fa.fit(Y.T)

# Plot the components
xs = np.asarray(ax.get_xlim())
w0, w1 = fa.components_[0]
ys = w1/w0*xs
ax.plot(xs, ys, color='r')
