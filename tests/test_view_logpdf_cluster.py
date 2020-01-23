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

from __future__ import division
from past.utils import old_div
import pytest

import numpy as np

from cgpm.mixtures.view import View
from cgpm.utils import general as gu


def retrieve_view():
    data = np.asarray([
        [1.1,   -2.1,    0],  # rowid=0
        [2.,      .1,    0],  # rowid=1
        [1.5, np.nan,   .5],  # rowid=2
        [4.7,    7.4,   .5],  # rowid=3
        [5.2,    9.6,   np.nan],  # rowid=4
    ])

    outputs = [0,1,2,]

    return View(
        {c: data[:,i].tolist() for i, c in enumerate(outputs)},
        outputs=[1000] + outputs,
        alpha=2.,
        cctypes=['normal'] * len(outputs),
        Zr=[0,0,0,1,1,]
    )


def test_crp_prior_logpdf():
    view = retrieve_view()
    crp_normalizer = view.alpha() + 5.
    cluster_logps = np.log(np.asarray([
        old_div(3, crp_normalizer),
        old_div(2, crp_normalizer),
        old_div(view.alpha(), crp_normalizer)
    ]))
    # Test the crp probabilities agree for a hypothetical row.
    for k in [0,1,2]:
        expected_logpdf = cluster_logps[k]
        crp_logpdf = view.crp.clusters[0].logpdf(None, {view.outputs[0]: k})
        assert np.allclose(expected_logpdf, crp_logpdf)
        view_logpdf = view.logpdf(None, {view.outputs[0]: k})
        assert np.allclose(view_logpdf, crp_logpdf)


def test_crp_posterior_logpdf():
    view = retrieve_view()
    fresh_row = {0:2, 1:3, 2:.5}
    logps = [
        view.logpdf(None, {view.outputs[0]: k}, fresh_row)
        for k in [0,1,2]
    ]
    assert np.allclose(gu.logsumexp(logps), 0)


def test_logpdf_observed_nan():
    view = retrieve_view()
    logp_view = view.logpdf(2, {1:1})
    logp_dim = view.dims[1].logpdf(2, {1:1}, None, {view.outputs[0]: view.Zr(2)})
    assert np.allclose(logp_view, logp_dim)


def test_logpdf_chain():
    view = retrieve_view()
    logp_cluster = view.logpdf(None, {view.outputs[0]: 0})
    logp_data = view.logpdf(None, {1:1, 2:0}, {view.outputs[0]: 0})
    logp_joint = view.logpdf(None, {1:1, 2:0, view.outputs[0]: 0})
    assert np.allclose(logp_cluster+logp_data, logp_joint)


def test_logpdf_bayes():
    view = retrieve_view()
    logp_posterior = view.logpdf(None, {view.outputs[0]: 0, 1:1}, {2:0})
    logp_evidence = view.logpdf(None, {2:0})
    logp_joint = view.logpdf(None, {1:1, 2:0, view.outputs[0]: 0})
    assert np.allclose(logp_joint - logp_evidence, logp_posterior)
