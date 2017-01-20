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

import cgpm.utils.test as tu

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, deep_merged


# ----- HELPER FUNCTIONS ----- #
def load_animals_engine():
    with open('tests/resources/animals_engine.pkl', 'rb') as f:
        animals_engine = Engine.from_pickle(f)
    return animals_engine
                         
                         
# ----- GLOBAL VARIABLES ----- #
simple_engine = tu.create_simple_engine()
simple_state = tu.create_simple_state()
simple_view = tu.create_simple_view()

animals_engine = load_animals_engine()
animals_state = animals_engine.get_state(4)
animals_view = animals_state.views[65]

Z = simple_view.exposed_latent
context_simple = 0
context_animals = 1


# ----- TEST POSTERIOR SCORE ----- #
def check_posterior_score_answer(cgpm, answer, target, query, context):
    kwargs = dict(target=target, query=query, debug=True)
    if isinstance(cgpm, (State, Engine)):
        kwargs['context'] = context
    
    s = cgpm.posterior_relevance_score(**kwargs)
    assert np.allclose(answer, s)

def logsumexp_conditional_densities_view(num_of_clusters, target, query):
    # logsumexp_k logpdf(clusters = k | target, query)
    view = simple_view
    evidence = deep_merged(target, query)
    rowids = evidence.keys()
    
    s = -np.inf
    for k in xrange(num_of_clusters+1):
        query = {r: {Z: k} for r in rowids}
        l = view.logpdf_set(query, evidence)
        s = logsumexp((s, l))
    return s
    
@pytest.mark.parametrize('cgpm', [simple_state, simple_view, simple_engine])
def test_value_one_hypothetical_one_nonhypothetical_rows(cgpm):
    # score({0: {0: 1}}; {1: {0: 1}}) = 1 - 24./56
    target = {0: {0: 1}}
    query = {1: {0: 1}}
    answer = 32./56
    check_posterior_score_answer(cgpm, answer, target, query, context_simple)

@pytest.mark.parametrize('cgpm', [simple_state, simple_view, simple_engine])
def test_value_two_hypothetical_rows(cgpm):
    # score({1: {0: 1}}; {2: {0: 1}})
    target = {1: {0: 1}}
    query = {2: {0: 1}}
    
    # compute answer based on logpdf_set
    log_answer = logsumexp_conditional_densities_view(1, target, query)
    check_posterior_score_answer(
        cgpm, np.exp(log_answer), target, query, context_simple)

@pytest.mark.parametrize('cgpm', [simple_state, simple_view, simple_engine])
def test_value_three_hypothetical_rows(cgpm):
    # score({1: {0: 1}}; {2: {0: 1}, 3: {0: 1}})
    target = {1: {0: 1}}
    query = {2: {0: 1}, 3: {0: 1}}

    # compute answer based on logpdf_set
    num = logsumexp_conditional_densities_view(1, target, query)
    den = logsumexp_conditional_densities_view(1, query, {})
    log_answer = num - den
    check_posterior_score_answer(
        cgpm, np.exp(log_answer), target, query, context_simple)


# ----- TEST COMMUTATIVITY ----- #
def check_commutativity(cgpm, target, query, context):
    debug = True
    if isinstance(cgpm, View):
        assert np.allclose(
            cgpm.posterior_relevance_score(target, query, debug),
            cgpm.posterior_relevance_score(query, target, debug),
            atol=1e-4)
    
    elif isinstance(cgpm, (State, Engine)):
        assert np.allclose(
            cgpm.posterior_relevance_score(target, query, context, debug),
            cgpm.posterior_relevance_score(query, target, context, debug),
            atol=1e-3)
    
    else:
        assert False
    
@pytest.mark.parametrize('cgpm', [simple_state, simple_view, simple_engine])
def test_commutativity_simple_one_hypothetical_one_nonhypothetical_rows(cgpm):
    check_commutativity(cgpm, {0: {0: 1}}, {1: {0: 0}}, context_simple)

@pytest.mark.parametrize('cgpm', [simple_state, simple_view, simple_engine])
def test_commutativity_simple_two_hypothetical_rows(cgpm):
    check_commutativity(cgpm, {1: {0: 1}}, {2: {0: 0}}, context_simple)
                                                       
# DEACTIVATED - Score is only commutative for single row queries
# @pytest.mark.parametrize('cgpm', [simple_state, simple_view, simple_engine])
# def test_commutativity_simple_three_hypothetical_rows(cgpm):
#     check_commutativity(
#         cgpm, {1: {0: 1}}, {2: {0: 0}, 3: {0: 1}}, context_simple)

@pytest.mark.parametrize('cgpm', [animals_state, animals_view, animals_engine])
def test_commutativity_animals_0_4(cgpm):
    check_commutativity(cgpm, {0: {}}, {4: {}}, context_animals)
                                               
@pytest.mark.parametrize('cgpm', [animals_state, animals_view, animals_engine])
def test_commutativity_animals_0_26(cgpm):
    check_commutativity(cgpm, {0: {}}, {26: {}}, context_animals)

# DEACTIVATED - Score is only commutative for single row queries
# @pytest.mark.parametrize('cgpm', [animals_state, animals_view, animals_engine])
# def test_commutativity_animals_0_4_26(cgpm):
#     check_commutativity(cgpm, {0: {}}, {4: {}, 26: {}}, context_animals)


# ----- TEST AGREEMENT BETWEEN VIEW AND STATE ----- #
def check_agreement_view_state(view, state, target, query, context):
    debug = True
    assert (view.posterior_relevance_score(target, query, debug) ==
            state.posterior_relevance_score(target, query, context, debug))

def test_agreement_view_simple_one_hypothetical_one_nonhypothetical_rows():
    check_agreement_view_state(
        simple_view, simple_state, {0: {0: 1}}, {1: {0: 0}}, context_simple)

def test_agreement_view_simple_two_hypothetical_rows():
    check_agreement_view_state(
        simple_view, simple_state, {1: {0: 1}}, {2: {0: 0}}, context_simple)

def test_agreement_view_simple_three_hypothetical_rows():
    check_agreement_view_state(
        simple_view, simple_state, {1: {0: 1}}, {2: {0: 0}, 3: {0: 1}},
        context_simple)

def test_agreement_view_animals_0_4():
    check_agreement_view_state(
        animals_view, animals_state, {0: {}}, {4: {}}, context_animals)
                                               
def test_agreement_view_animals_0_26():
    check_agreement_view_state(
        animals_view, animals_state, {0: {}}, {26: {}}, context_animals)

def test_agreement_view_animals_0_4_26():
    check_agreement_view_state(
        animals_view, animals_state, {0: {}}, {4: {}, 26: {}}, context_animals)
