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

import time

import pytest

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.dummy.fourway import FourWay
from cgpm.dummy.twoway import TwoWay
from cgpm.utils import general as gu


def test_all_kernels():
    rng = gu.gen_rng(0)
    X = rng.normal(size=(5,5))
    state = State(X, cctypes=['normal']*5)
    state.transition(N=5)
    for k, n in state.to_metadata()['diagnostics']['iterations'].iteritems():
        assert n == 5

def test_individual_kernels():
    rng = gu.gen_rng(0)
    X = rng.normal(size=(5,5))
    state = State(X, cctypes=['normal']*5)
    state.transition(N=3, kernels=['alpha', 'rows'])
    check_expected_counts(
        state.diagnostics['iterations'],
        {'alpha':3, 'rows':3})
    state.transition(N=5, kernels=['view_alphas', 'column_params'])
    check_expected_counts(
        state.to_metadata()['diagnostics']['iterations'],
        {'alpha':3, 'rows':3, 'view_alphas':5, 'column_params':5})
    state.transition(
        N=1, kernels=['view_alphas', 'column_params', 'column_hypers'])
    check_expected_counts(
        state.to_metadata()['diagnostics']['iterations'],
        {'alpha':3, 'rows':3, 'view_alphas':6, 'column_params':6,
        'column_hypers':1})


def test_transition_foreign():
    rng = gu.gen_rng(0)
    X = rng.normal(size=(5,5))
    state = State(X, cctypes=['normal']*5)

    token_a = state.compose_cgpm(FourWay(outputs=[12], inputs=[0,1], rng=rng))
    state.transition_foreign(cols=[12], N=5)
    check_expected_counts(
        state.diagnostics['iterations'],
        {'foreign-%s'%token_a: 5})

    token_b = state.compose_cgpm(TwoWay(outputs=[22], inputs=[2], rng=rng))
    state.transition_foreign(cols=[22], N=1)
    check_expected_counts(
        state.diagnostics['iterations'],
        {'foreign-%s'%token_a: 5, 'foreign-%s'%token_b: 1})


    state.transition_foreign(N=3)
    check_expected_counts(
        state.diagnostics['iterations'],
        {'foreign-%s'%token_a: 8, 'foreign-%s'%token_b: 4})

    start = time.time()
    state.transition_foreign(S=2)
    assert time.time() - start >= 2

    # Crash test for engine to transition everyone.
    engine = Engine(X, cctypes=['normal']*5)
    engine.compose_cgpm([FourWay(outputs=[12], inputs=[0,1], rng=rng)])
    engine.transition(N=4)
    # Cannot use transition with a foreign variable.
    with pytest.raises(ValueError):
        engine.transition(N=4, cols=state.outputs+[12], multiprocess=False)
    # Cannot use transition_foreign with a Crosscat variable.
    with pytest.raises(ValueError):
        engine.transition_foreign(cols=[0, 12], N=4, multiprocess=False)
    engine.transition_foreign(cols=[12], N=4)


def check_expected_counts(actual, expected):
    for k, n in expected.iteritems():
        assert n == actual[k]
