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

from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, deep_merged

# # Helper Functions
def initialize_view():
    data = np.array([[1, 1]])
    D = len(data[0])
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(
        X,
        outputs=[1000] + outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        hypers={i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zr=[0])
    return view

Z = initialize_view().exposed_latent

def check_posterior_score_answer(answer, target_row, query_row_set):
    view = initialize_view()
    s = view.posterior_relevance_score(target_row, query_row_set)
    assert np.allclose(answer, s)

def logsumexp_conditional_densities(num_of_clusters, target, query):
    # logsumexp_k logpdf(clusters = k | target, query)
    view = initialize_view()
    evidence = deep_merged(target, query)
    rowids = evidence.keys()
    
    s = -np.inf
    for k in xrange(num_of_clusters+1):
        query = {r: {Z: k} for r in rowids}
        l = view.logpdf_set(query, evidence)
        s = logsumexp((s, l))
    
    return s
    
def test_one_hypothetical_one_nonhypothetical_rows():
    # score({0: {0: 1}}; {1: {0: 1}}) = 1 - 24./56
    target = {0: {0: 1}}
    query = {1: {0: 1}}
    answer = np.log(32./56)
    check_posterior_score_answer(answer, target, query)

def test_two_hypothetical_rows():
    # score({1: {0: 1}}; {2: {0: 1}})
    target = {1: {0: 1}}
    query = {2: {0: 1}}
    
    # compute answer based on logpdf_set
    answer = logsumexp_conditional_densities(1, target, query)
    check_posterior_score_answer(answer, target, query)

def test_three_hypothetical_rows():
    # score({1: {0: 1}}; {2: {0: 1}, 3: {0: 1}})
    target = {1: {0: 1}}
    query = {2: {0: 1}, 3: {0: 1}}

    # compute answer based on logpdf_set
    num = logsumexp_conditional_densities(1, target, query)
    den = logsumexp_conditional_densities(1, query, {})
    answer = num - den
    check_posterior_score_answer(answer, target, query)
