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

from gpmcc.utils import config as cu


test_inputs = {
    'bernoulli'         :
        (None, [0, 1.], [-1, .5, 3]),
    'beta_uc'           :
        (None, [.3, .1, .9], [-1, 1.02, 21]),
    'categorical'       :
        ({'k':4}, [0., 1, 2, 3.], [-1, 2.5, 4]),
    'exponential'       :
        (None, [0, 1, 2, 3], [-1, -2.5]),
    'geometric'         :
        (None, [0, 2, 12], [-1, .5, -4]),
    'lognormal'         :
        (None, [1, 2, 3], [-12, -0.01, 0]),
    'normal'            :
        (None, [-1, 0, 10], []),
    'normal_trunc'      :
        ({'l':-1,'h':10}, [0, 4, 9], [44,-1.02]),
    'poisson'           :
        (None, [0, 5, 11], [-1, .5, -4]),
    'vonmises'          :
        (None, [0.1, 3.14, 6.2], [-1, 7, 12])
}

def test_distributions():
    for cctype, params in test_inputs.iteritems():
        distargs, good, bad = params
        assert_distribution(cctype, distargs, good, bad)

def assert_distribution(cctype, distargs, good, bad):
    def test_good(g):
        print cctype, g
        model.incorporate(g)
        model.unincorporate(g)
        assert model.logpdf(g) != -float('inf')
    def test_bad(b):
        with pytest.raises(ValueError):
            print cctype, b
            model.incorporate(b)
        with pytest.raises(ValueError):
            model.unincorporate(b)
        assert model.logpdf(b) == -float('inf')
    model = cu.cctype_class(cctype)(distargs=distargs)
    for g in good:
        test_good(g)
    for b in bad:
        test_bad(b)
