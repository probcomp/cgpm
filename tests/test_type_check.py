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

from collections import namedtuple

import pytest

from cgpm.utils import config as cu


Case = namedtuple(
    'Case', ['outputs', 'inputs', 'distargs', 'good', 'bad'])

cases = {
    'bernoulli' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[0, 1.],
        bad=[-1, .5, 3]),

    'beta' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[.3, .1, .9, 0.001, .9999],
        bad=[-1, 1.02, 21]),

    'categorical' : Case(
        outputs=[0],
        inputs=None,
        distargs={'k': 4},
        good=[0., 1, 2, 3.],
        bad=[-1, 2.5, 4]),

    'exponential' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[0, 1, 2, 3],
        bad=[-1, -2.5]),

    'geometric' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[0, 2, 12],
        bad=[-1, .5, -4]),

    'lognormal' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[1, 2, 3],
        bad=[-12, -0.01, 0]),

    'normal' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[-1, 0, 10],
        bad=[]),

    'normal_trunc' : Case(
        outputs=[0],
        inputs=None,
        distargs={'l': -1, 'h': 10},
        good=[0, 4, 9],
        bad=[44, -1.02]),

    'poisson' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[0, 5, 11],
        bad=[-1, .5, -4]),

    'random_forest' : Case(
        outputs=[0],
        inputs=[1, 2],
        distargs={'k': 2, 'inputs':{'stattypes': [1,2]}},
        good=[(0, {1:1, 2:2}), (1, {1:0, 2:2})],
        bad=[(-1, {1:1, 2:2}), (0, {0:1, 2:2}), (0, {1: 3})]),

    'vonmises' : Case(
        outputs=[0],
        inputs=None,
        distargs=None,
        good=[0.1, 3.14, 6.2],
        bad=[-1, 7, 12]),

    'linear_regression' : Case(
        outputs=[0],
        inputs=[1, 2],
        distargs={
            'inputs': {
                'stattypes': ['normal', 'bernoulli'],
                'statargs': [None, {'k':2}]}},
        good=[(0, {1:1, 2:0})],
        bad=[(0, {0:1, 1:1, 2:0})]),
}


def get_observation_inputs(t):
    # Assumes that the output is always column id 0.
    return ({0: t[0]}, t[1]) if isinstance(t, tuple) else ({0: t}, None)


@pytest.mark.parametrize('cctype', cases.keys())
def test_distributions(cctype):
    case = cases[cctype]
    assert_distribution(
        cctype, case.outputs, case.inputs, case.distargs, case.good, case.bad)


def assert_distribution(cctype, outputs, inputs, distargs, good, bad):
    model = cu.cctype_class(cctype)(outputs, inputs, distargs=distargs)
    for rowid, g in enumerate(good):
        assert_good(model, rowid, g)
    for rowid, b in enumerate(bad):
        assert_bad(model, rowid, b)


def assert_good(model, rowid, g):
    observation, inputs = get_observation_inputs(g)
    model.incorporate(rowid, observation, inputs)
    model.unincorporate(rowid)
    assert model.logpdf(-1, observation, None, inputs) != -float('inf')


def assert_bad(model, rowid, b):
    observation, inputs = get_observation_inputs(b)
    with pytest.raises(Exception):
        model.incorporate(rowid, observation, inputs)
    with pytest.raises(Exception):
        model.unincorporate(rowid)
    try: # GPM return negative infinity for invalid input.
        assert model.logpdf(-1, observation, None, inputs) == -float('inf')
    except Exception: # Conditional GPM throws error on wrong input variables.
        assert True
