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


cctypes_distargs_good_bad = {
    'bernoulli'         : (None, [0, 1.], [-1, .5, 3]),
    'beta_uc'           : (None, [.3, .1, .9], [-1, 1.02, 21]),
    'categorical'       : ({'k':4}, [0., 1, 2, 3.], [-1, 2.5, 4]),
    'exponential'       : (None, [0, 1, 2, 3], [-1, -2.5]),
    'geometric'         : (None, [0, 2, 12], [-1, .5, -4]),
    'lognormal'         : (None, [1, 2, 3], [-12, -0.01, 0]),
    'normal'            : (None, [-1, 0, 10], []),
    'normal_trunc'      : ({'l':-1,'h':10}, [0, 4, 9], [44,-1.02]),
    'poisson'           : (None, [0, 5, 11], [-1, .5, -4]),
    'random_forest'     : ({'k':1, 'cctypes':[0,1]}, [(0,[1,2])], [(-1,[1,2])]),
    'vonmises'          : (None, [0.1, 3.14, 6.2], [-1, 7, 12])
}


@pytest.mark.parametrize('cctype', cctypes_distargs_good_bad.keys())
def test_distributions(cctype):
    (distargs, good, bad) = cctypes_distargs_good_bad[cctype]
    assert_distribution(cctype, distargs, good, bad)


def assert_distribution(cctype, distargs, good, bad):
    model = cu.cctype_class(cctype)(distargs=distargs)
    for g in good:
        assert_good(model, g)
    for b in bad:
        assert_bad(model, b)


def assert_good(model, g):
    (x, y) = (g[0], g[1]) if isinstance(g, tuple) else (g, None)
    model.incorporate(x, y)
    model.unincorporate(x, y)
    assert model.logpdf(x, y) != -float('inf')


def assert_bad(model, b):
    (x, y) = (b[0], b[1]) if isinstance(b, tuple) else (b, None)
    with pytest.raises(ValueError):
        model.incorporate(x, y)
    with pytest.raises(ValueError):
        model.unincorporate(x, y)
    assert model.logpdf(x, y) == -float('inf')
