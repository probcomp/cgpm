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

import unittest

from gpmcc.state import State
from gpmcc.utils import general as gu


def test_all_kernels():
    rng = gu.gen_rng(0)
    X = rng.normal(size=(5,5))
    state = State(X, cctypes=['normal']*5)
    state.transition(N=5)
    for k, n in state.to_metadata()['iterations'].iteritems():
        assert n == 5

def test_individual_kernels():
    rng = gu.gen_rng(0)
    X = rng.normal(size=(5,5))
    state = State(X, cctypes=['normal']*5)
    state.transition(N=3, kernels=['alpha', 'rows'])
    check_expected_counts(state.iterations, {'alpha':3, 'rows':3})
    state.transition(N=5, kernels=['view_alphas', 'column_params'])
    check_expected_counts(
        state.to_metadata()['iterations'],
        {'alpha':3, 'rows':3, 'view_alphas':5, 'column_params':5})
    state.transition(
        N=1, kernels=['view_alphas', 'column_params', 'column_hypers'])
    check_expected_counts(
        state.to_metadata()['iterations'],
        {'alpha':3, 'rows':3, 'view_alphas':6, 'column_params':6,
        'column_hypers':1})

def check_expected_counts(actual, expected):
    for k, n in actual.iteritems():
        if k in expected:
            assert n == expected[k]
        else:
            assert n == 0

if __name__ == '__main__':
    unittest.main()
