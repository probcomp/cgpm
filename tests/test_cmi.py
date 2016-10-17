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


import numpy as np
import pytest

from cgpm.crosscat.state import State
from cgpm.utils.general import gen_rng


def test_entropy_bernoulli__ci_():
    rng = gen_rng(10)
    T = rng.choice([0,1], p=[.3,.7], size=250).reshape(-1,1)
    state = State(T, cctypes=['bernoulli'], rng=rng)
    state.transition(N=30)
    # Exact computation.
    logp = state.logpdf_bulk([-1,-1], [{0:0}, {0:1}])
    entropy_exact = -np.sum(np.exp(logp)*logp)
    # Monte Carlo computation.
    entropy_mc = state.mutual_information(0, 0, N=1000)
    # Punt CLT analysis and go for 1 dp.
    assert np.allclose(entropy_exact, entropy_mc, atol=.1)


def test_cmi_different_views__ci_():
    rng = gen_rng(0)
    T = np.zeros((50,3))
    T[:,0] = rng.normal(loc=-5, scale=1, size=50)
    T[:,1] = rng.normal(loc=2, scale=2, size=50)
    T[:,2] = rng.normal(loc=12, scale=3, size=50)
    state = State(
        T, outputs=[0, 1, 2], cctypes=['normal','normal','normal'],
        Zv={0:0, 1:1, 2:2}, rng=rng)
    state.transition(N=30, kernels=['alpha', 'view_alphas',
        'column_params', 'column_hypers','rows'])

    mi01 = state.mutual_information(0, 1)
    mi02 = state.mutual_information(0, 2)
    mi12 = state.mutual_information(1, 2)

    # Marginal MI all zero.
    assert np.allclose(mi01, 0)
    assert np.allclose(mi02, 0)
    assert np.allclose(mi12, 0)

    # CMI on variable in other view equal to MI.
    assert np.allclose(
        state.mutual_information(0, 1, evidence={2:10}), mi01)
    assert np.allclose(
        state.mutual_information(0, 2, evidence={1:0}), mi02)
    assert np.allclose(
        state.mutual_information(1, 2, evidence={0:-2}), mi12)
    assert np.allclose(
        state.mutual_information(1, 2, evidence={0:None}, T=5), mi12)

def test_cmi_marginal_crash():
    X = np.eye(5)
    cctypes = ['normal'] * 5
    s = State(X, cctypes=cctypes)
    s.transition(N=2)
    # One marginalized evidence variable.
    s.mutual_information(0, 1, {2:None}, T=10, N=10)
    # Two marginalized evidence variables.
    s.mutual_information(0, 1, {2:None, 3:None}, T=10, N=10)
    # Two marginalized evidence variables and one constrained variable.
    s.mutual_information(0, 1, {2:None, 3:None, 4:0}, T=10, N=10)
