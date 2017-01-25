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

"""
Find the math for these tests in
https://docs.google.com/document/d/15_JGb39TuuSup_0gIBJTuMHYs8HS4m_TzjJ-AOXnh9M/edit
"""

import numpy as np
import pytest

import cgpm.utils.test as tu

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State

# ----- GLOBAL VARIABLES ----- #
simple_engine = tu.create_simple_engine()
simple_state = tu.create_simple_state()
simple_view = tu.gen_simple_view()
simple_cgpms = [simple_view]

# -If I decide to implement logpdf_set on state and engine later,
#  I should uncomment the line below for the first tests.
# simple_cgpms = [simple_view, simple_state, simple_engine]

Z = simple_view.exposed_latent
context_simple = 0

# ----- TEST ANALYTICAL LOGPDF-SET ----- #
def check_logpdf_set_answer(cgpm, answer, query, context, evidence=None):
    kwargs = dict(query=query, evidence=evidence, debug=True)
    if isinstance(cgpm, (State, Engine)):
        kwargs['context'] = context
    
    l = cgpm.logpdf_set(query, evidence, debug=True)
    s = np.mean(l)  # if Engine, average logpdf of each state
    assert np.allclose(answer, s)

# ----- TEST MARGINAL ----- #
@pytest.mark.parametrize('cgpm', simple_cgpms)
def test_marginal_one_hypothetical_query_row(cgpm):
    # P(x[1,0] = 1) = 7./12
    query = {1: {0: 1}}
    answer = np.log(7./12)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_marginal_one_nonhypothetical_query_row(cgpm):
    # P(x[0,0] = 1) = 2./3
    query = {0: {0: 1}}
    answer = np.log(1./2)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_marginal_one_hypothetical_cluster(cgpm):
    # P(Z[1] = 0) = 1./2
    query = {1: {Z: 0}}
    answer = np.log(1./2)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_marginal_one_nonhypothetical_cluster(cgpm):
    # P(Z[0] = 0) = 1.
    query = {0: {Z: 0}}
    answer = np.log(1)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

# ----- TEST JOINT DENSITY  ----- #
@pytest.mark.parametrize('cgpm', simple_cgpms)
def test_joint_two_rows_one_column(cgpm):
    # P(x[0,0] = 1, x[1,0] = 1) = 7./24
    query = {0: {0: 1}, 1: {0: 1}}
    answer = np.log(7./24)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_joint_two_rows_two_columns(cgpm):
    # P(x[0,:] = [1,1], x[1,:] = [1,1]) = 25./288
    query = {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}}
    answer = np.log(25./288)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_joint_two_rows_two_columns_missing_values(cgpm):
    # P(x[0,0] = 1, x[1,1]=1)
    query = {0: {0: 1}, 1: {1: 1}}
    answer = np.log(1./4)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_joint_two_rows_with_clusters(cgpm):
    # P(row 0: {0: 1, Z: 0}, row 1: {0: 1, Z: 1}) = 1./8
    query = {0: {0: 1, Z: 0}, 1: {0: 1, Z: 1}}
    answer = np.log(1./8)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_joint_two_clusters(cgpm):
    # p({0: {Z: 0}, 1: {Z: 1}}) = 1./2
    query = {0: {Z: 0}, 1: {Z: 1}}
    answer = np.log(1./2)
    check_logpdf_set_answer(cgpm, answer, query, context_simple)

# ----- TEST CONDITIONAL DENSITY ----- #
@pytest.mark.parametrize('cgpm', simple_cgpms)
def test_conditional_two_rows_given_two_clusters(cgpm):
    # p({row 0: {0: 1}, row 1: {0: 1}} | {0: {Z: 0}, 1: {Z: 1}}) = 1./4
    query = {0: {0: 1}, 1: {0: 1}}
    evidence = {0: {Z: 0}, 1: {Z: 1}}
    answer = np.log(1./4)
    check_logpdf_set_answer(cgpm, answer, query, context_simple, evidence)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_conditional_two_clusters_given_two_rows(cgpm):
    # p({0: {Z: 0}, 1: {Z: 1}} | {row 0: {0: 1}, row 1: {0: 1}}) = 1./8 * 24./7
    query = {0: {Z: 0}, 1: {Z: 1}}
    evidence = {0: {0: 1}, 1: {0: 1}}
    answer = np.log(24./56) 
    check_logpdf_set_answer(cgpm, answer, query, context_simple, evidence)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_conditional_one_column_one_row_given_another_row_first_part(cgpm):
    # P(x[1,0] = 1 | x[0,0] = 1) = 7./12
    # Missing column and non-hypothetical row
    query = {1: {0: 1}}
    evidence = {0: {0: 1}}
    answer = np.log(7./12)
    check_logpdf_set_answer(cgpm, answer, query, context_simple, evidence)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_conditional_one_column_one_row_given_another_row_second_part(cgpm):
    # P(x[0,0] = 1 | x[1,0] = 1) = 7./12
    query = {0: {0: 1}}
    evidence = {1: {0: 1}}
    answer = np.log(7./12)
    check_logpdf_set_answer(cgpm, answer, query, context_simple, evidence)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_conditional_two_columns_one_row_given_another_row_first_part(cgpm):
    # P(x[1,:] = [1,1] | x[0,:] = [1,1]) = 25./72
    query = {1: {0: 1, 1: 1}}
    evidence = {0: {0: 1, 1: 1}}
    answer = np.log(25./72)
    check_logpdf_set_answer(cgpm, answer, query, context_simple, evidence)

@pytest.mark.parametrize('cgpm', simple_cgpms)    
def test_conditional_two_columns_one_row_given_another_row_second_part(cgpm):
    # P(x[0,:] = [1,1] | x[1,:] = [1,1]) = 25./72
    query = {0: {0: 1, 1: 1}}
    evidence = {1: {0: 1, 1: 1}}
    answer = np.log(25./72)
    check_logpdf_set_answer(cgpm, answer, query, context_simple, evidence)
