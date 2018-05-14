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

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.utils.general import gen_rng

from markers import integration


def test_entropy_bernoulli_univariate__ci_():
    rng = gen_rng(10)

    # Generate a univariate Bernoulli dataset.
    T = rng.choice([0,1], p=[.3,.7], size=250).reshape(-1,1)

    engine = Engine(T, cctypes=['bernoulli'], rng=rng, num_states=16)
    engine.transition(S=15)

    # exact computation.
    entropy_exact = - (.3*np.log(.3) + .7*np.log(.7))

    # logpdf computation.
    logps = engine.logpdf_bulk([-1,-1], [{0:0}, {0:1}])
    entropy_logpdf = [-np.sum(np.exp(logp)*logp) for logp in logps]

    # mutual_information computation.
    entropy_mi = engine.mutual_information([0], [0], N=1000)

    # Punt CLT analysis and go for 1 dp.
    assert np.allclose(entropy_exact, entropy_logpdf, atol=.1)
    assert np.allclose(entropy_exact, entropy_mi, atol=.1)
    assert np.allclose(entropy_logpdf, entropy_mi, atol=.05)

@integration
def test_entropy_bernoulli_bivariate__ci_():
    rng = gen_rng(10)

    # Generate a bivariate Bernoulli dataset.
    PX = [.3, .7]
    PY = [[.2, .8], [.6, .4]]
    TX = rng.choice([0,1], p=PX, size=250)
    TY = np.zeros(shape=len(TX))
    TY[TX==0] = rng.choice([0,1], p=PY[0], size=len(TX[TX==0]))
    TY[TX==1] = rng.choice([0,1], p=PY[1], size=len(TX[TX==1]))
    T = np.column_stack((TY,TX))

    engine = Engine(
        T,
        cctypes=['categorical', 'categorical'],
        distargs=[{'k':2}, {'k':2}],
        num_states=64,
        rng=rng,
    )

    engine.transition_lovecat(N=200)

    # exact computation
    entropy_exact = (
        - PX[0]*PY[0][0] * np.log(PX[0]*PY[0][0])
        - PX[0]*PY[0][1] * np.log(PX[0]*PY[0][1])
        - PX[1]*PY[1][0] * np.log(PX[1]*PY[1][0])
        - PX[1]*PY[1][1] * np.log(PX[1]*PY[1][1])
    )

    # logpdf computation
    logps = engine.logpdf_bulk(
        [-1,-1,-1,-1], [{0:0, 1:0}, {0:0, 1:1}, {0:1, 1:0}, {0:1, 1:1}]
    )
    entropy_logpdf = [-np.sum(np.exp(logp)*logp) for logp in logps]

    # mutual_information computation.
    entropy_mi = engine.mutual_information([0,1], [0,1], N=1000)

    # Punt CLT analysis and go for a small tolerance.
    assert np.allclose(entropy_exact, entropy_logpdf, atol=.1)
    assert np.allclose(entropy_exact, entropy_mi, atol=.1)
    assert np.allclose(entropy_logpdf, entropy_mi, atol=.1)

def test_cmi_different_views__ci_():
    rng = gen_rng(0)
    T = np.zeros((50,3))
    T[:,0] = rng.normal(loc=-5, scale=1, size=50)
    T[:,1] = rng.normal(loc=2, scale=2, size=50)
    T[:,2] = rng.normal(loc=12, scale=3, size=50)
    state = State(
        T,
        outputs=[0, 1, 2],
        cctypes=['normal','normal','normal'],
        Zv={0:0, 1:1, 2:2},
        rng=rng
    )
    state.transition(N=30,
        kernels=['alpha','view_alphas','column_params','column_hypers','rows'])

    mi01 = state.mutual_information([0], [1])
    mi02 = state.mutual_information([0], [2])
    mi12 = state.mutual_information([1], [2])

    # Marginal MI all zero.
    assert np.allclose(mi01, 0)
    assert np.allclose(mi02, 0)
    assert np.allclose(mi12, 0)

    # CMI on variable in other view equal to MI.
    assert np.allclose(state.mutual_information([0], [1], {2:10}), mi01)
    assert np.allclose(state.mutual_information([0], [2], {1:0}), mi02)
    assert np.allclose(state.mutual_information([1], [2], {0:-2}), mi12)
    assert np.allclose(state.mutual_information([1], [2], {0:None}, T=5), mi12)

def test_cmi_marginal_crash():
    X = np.eye(5)
    cctypes = ['normal'] * 5
    s = State(X, Zv={0:0, 1:0, 2:0, 3:1, 4:1}, cctypes=cctypes)
    # One marginalized constraint variable.
    s.mutual_information([0], [1], {2:None}, T=10, N=10)
    # Two marginalized constraint variables.
    s.mutual_information([0], [1], {2:None, 3:None}, T=10, N=10)
    # Two marginalized constraint variables and one constrained variable.
    s.mutual_information([0], [1], {2:None, 3:None, 4:0}, T=10, N=10)

def test_cmi_multivariate_crash():
    X = np.eye(5)
    cctypes = ['normal'] * 5
    s = State(X, Zv={0:0, 1:0, 2:0, 3:1, 4:1}, cctypes=cctypes)
    s.mutual_information([0,1], [0,1], {2:1}, T=10, N=10)
    s.mutual_information([0,1], [0,1], {2:None}, T=10, N=10)
    s.mutual_information([2,4], [0,1,3], {}, T=10, N=10)
    # Duplicate in 2 query and constraint.
    with pytest.raises(ValueError):
        s.mutual_information([2,4], [1,3], {0:1, 2:None}, T=10, N=10)
    # Duplicate in 3 query.
    with pytest.raises(ValueError):
        s.mutual_information([2,3,4], [1,3], {0:None}, T=10, N=10)
