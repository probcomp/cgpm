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

from cgpm.mixtures.view import View
from cgpm.utils import general as gu


def retrieve_view():
    data = np.asarray([
        [1.1,   -2.1,    0],  # rowid=0
        [2.,      .1,    0],  # rowid=1
        [1.5,      1,   .5],  # rowid=2
        [4.7,    7.4,   .5],  # rowid=3
        [5.2,    9.6,   .5],  # rowid=4
    ])

    outputs = [0,1,2,]

    return View(
        {c: data[:,i].tolist() for i, c in enumerate(outputs)},
        outputs=[1000] + outputs,
        alpha=2.,
        cctypes=['normal'] * len(outputs),
        Zr=[0,0,0,1,1,]
    )

def test_crp_logpdf():
    view = retrieve_view()

    crp_normalizer = view.alpha() + 5.
    cluster_logps = np.log(np.asarray([
        3 / crp_normalizer,
        2 / crp_normalizer,
        view.alpha() / crp_normalizer
    ]))

    # Test the crp probabilities agree for a hypothetical row.
    for k in [0,1,2]:
        expected_logpdf = cluster_logps[k]
        crp_logpdf = view.crp.clusters[0].logpdf(None, {view.outputs[0]: k})
        assert np.allclose(expected_logpdf, crp_logpdf)
        view_logpdf = view.logpdf(None, {view.outputs[0]: k})
        assert np.allclose(view_logpdf, crp_logpdf)

# math_out = np.log(1./3)
# test_out = view.crp.clusters[0].logpdf(rowid=0, query={1000: 0})
# assert math_out == test_out  # Passes

# # Same thing for view
# test_out = view.logpdf(rowid=0, query={1000: 1})
# assert math_out == test_out  # Fails with test_out == 0.0
