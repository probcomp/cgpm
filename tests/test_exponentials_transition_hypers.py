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

import numpy as np

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


cctypes = [
    ('normal', None),
    ('categorical', {'k':4}),
    ('lognormal', None),
    ('poisson', None),
    ('bernoulli', None),
    ('exponential', None),
    ('geometric', None),
    ('vonmises', None)
    ]


@pytest.mark.parametrize('cctype', cctypes)
def test_transition_hypers(cctype):
    name, arg = cctype
    model = cu.cctype_class(name)(
        outputs=[0], inputs=None, distargs=arg, rng=gu.gen_rng(10))
    D, Zv, Zc = tu.gen_data_table(
        50, [1], [[.33, .33, .34]], [name], [arg], [.8], rng=gu.gen_rng(1))

    hypers_previous = model.get_hypers()
    for rowid, x in enumerate(np.ravel(D)[:25]):
        model.incorporate(rowid, {0:x}, None)
    model.transition_hypers(N=3)
    hypers_new = model.get_hypers()
    assert not all(
        np.allclose(hypers_new[hyper], hypers_previous[hyper])
        for hyper in hypers_new)

    for rowid, x in enumerate(np.ravel(D)[:25]):
        model.incorporate(rowid+25, {0:x}, None)
    model.transition_hypers(N=3)
    hypers_newer = model.get_hypers()
    assert not all(
        np.allclose(hypers_new[hyper], hypers_newer[hyper])
        for hyper in hypers_newer)

    # In general inference should improve the log score.
    # logpdf_score = model.logpdf_score()
    # model.transition_hypers(N=200)
    # assert model.logpdf_score() > logpdf_score

