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

from cgpm.crosscat.state import State
from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, deep_merged

# # Helper Functions
def initialize_trivial_view():
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

def initialize_trivial_state():
    data = np.array([[1, 1, 1]])
    R = len(data)
    D = len(data[0])
    outputs = range(D)
    state = State(
        data,
        outputs=outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        distargs={i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zv={0: 0, 1: 0, 2: 1},
        view_alphas=[1.]*D,
        Zrv={0: [0]*R, 1: [0]*R})
    return state
        
def load_animals_state():
    with open('tests/resources/animals_state.pkl', 'rb') as f:
        animals_state = State.from_pickle(f)
    return animals_state

def load_animals_view():
    animals_state = load_animals_state()
    view = animals_state.views[65]
    return view

# ----- GLOBAL VARIABLES ----- #
trivial_state = initialize_trivial_state()
trivial_view = initialize_trivial_view()
animals_state = load_animals_state()
animals_view = load_animals_view()
Z = trivial_view.exposed_latent

def check_posterior_score_answer(cgpm, answer, target, query):
    kwargs = dict(target=target, query=query, debug=True)
    if isinstance(cgpm, State):
        kwargs['context'] = 0
    
    s = cgpm.posterior_relevance_score(**kwargs)
    assert np.allclose(answer, s)

def logsumexp_conditional_densities_view(num_of_clusters, target, query):
    # logsumexp_k logpdf(clusters = k | target, query)
    view = initialize_trivial_view()
    evidence = deep_merged(target, query)
    rowids = evidence.keys()
    
    s = -np.inf
    for k in xrange(num_of_clusters+1):
        query = {r: {Z: k} for r in rowids}
        l = view.logpdf_set(query, evidence)
        s = logsumexp((s, l))
    return s
    
@pytest.mark.parametrize('cgpm', [trivial_state, trivial_view])
def test_value_one_hypothetical_one_nonhypothetical_rows(cgpm):
    # score({0: {0: 1}}; {1: {0: 1}}) = 1 - 24./56
    target = {0: {0: 1}}
    query = {1: {0: 1}}
    answer = 32./56
    check_posterior_score_answer(cgpm, answer, target, query)

@pytest.mark.parametrize('cgpm', [trivial_state, trivial_view])
def test_value_two_hypothetical_rows(cgpm):
    # score({1: {0: 1}}; {2: {0: 1}})
    target = {1: {0: 1}}
    query = {2: {0: 1}}
    
    # compute answer based on logpdf_set
    log_answer = logsumexp_conditional_densities_view(1, target, query)
    check_posterior_score_answer(cgpm, np.exp(log_answer), target, query)

@pytest.mark.parametrize('cgpm', [trivial_state, trivial_view])
def test_value_three_hypothetical_rows(cgpm):
    # score({1: {0: 1}}; {2: {0: 1}, 3: {0: 1}})
    target = {1: {0: 1}}
    query = {2: {0: 1}, 3: {0: 1}}

    # compute answer based on logpdf_set
    num = logsumexp_conditional_densities_view(1, target, query)
    den = logsumexp_conditional_densities_view(1, query, {})
    log_answer = num - den
    check_posterior_score_answer(cgpm, np.exp(log_answer), target, query)

# ----- TEST COMMUTATIVITY ----- #
def check_commutativity(cgpm, target, query):
    debug = True
    if isinstance(cgpm, View):
        np.allclose(
            np.log(cgpm.posterior_relevance_score(target, query, debug)),
            np.log(cgpm.posterior_relevance_score(query, target, debug)))
    
    elif isinstance(cgpm, State):
        np.allclose(
            np.log(cgpm.posterior_relevance_score(target, query, debug)),
            np.log(cgpm.posterior_relevance_score(query, target, debug)))
    
    
@pytest.mark.parametrize('cgpm', [trivial_state, trivial_view])
def test_commutativity_trivial_one_hypothetical_one_nonhypothetical_rows(cgpm):
    check_commutativity(cgpm, {0: {0: 1}}, {1: {0: 0}})

@pytest.mark.parametrize('cgpm', [trivial_state, trivial_view])
def test_commutativity_trivial_two_hypothetical_rows(cgpm):
    check_commutativity(cgpm, {1: {0: 1}}, {2: {0: 0}})
                                                       
@pytest.mark.parametrize('cgpm', [trivial_state, trivial_view])
def test_commutativity_trivial_three_hypothetical_rows(cgpm):
    check_commutativity(cgpm, {1: {0: 1}}, {2: {0: 0}, 3: {0: 1}})

@pytest.mark.parametrize('cgpm', [animals_state, animals_view])
def test_commutativity_animals_0_4(cgpm):
    check_commutativity(cgpm, {0: {}}, {4: {}})
                                               
@pytest.mark.parametrize('cgpm', [animals_state, animals_view])
def test_commutativity_animals_0_26(cgpm):
    check_commutativity(cgpm, {0: {}}, {26: {}})

@pytest.mark.parametrize('cgpm', [animals_state, animals_view])
def test_commutativity_animals_0_4_26(cgpm):
    check_commutativity(cgpm, {0: {}}, {4: {}, 26: {}})

