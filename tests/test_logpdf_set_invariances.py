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

from itertools import product

import cgpm.utils.test as tu

from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, deep_merged


# ----- GLOBAL VARIABLES ----- #
simple_view = tu.create_simple_view()
multitype_view = tu.create_multitype_view()
test_cgpms = [simple_view, multitype_view]

# ----- TEST NULL QUERY ----- #
def check_null_query(cgpm, null_query):
    '''Check for P(null) = 1 or log P(null) = 0''' 
    l = cgpm.logpdf_set(null_query)
    assert np.allclose(0., l)

null_query_examples = [
    {},  # no rowid 
    {0: {}},  # non-hypothetical row
    {-1: {}},  # hypothetical row
    {-1: {}, 0: {}}  # joint row
]

@pytest.mark.parametrize('cgpm', [simple_view])
@pytest.mark.parametrize('null_query', null_query_examples)
def test_null_query_simple(cgpm, null_query):
    check_null_query(cgpm, null_query)

# ----- TEST NULL EVIDENCE ----- #
# P({x} | null) = P(x)

# ----- TEST EXTREME DISTARGS ----- #
# P({x} | {y}, distargs=extreme) = P(x)

# ----- TEST EXTREME CRP HYPERS ----- #
# P({x} | {y}


# TEMPORARILY DEACTIVATED
# ----- TEST PRODUCT RULE ----- #
# logpdf-set(query, evidence) = 
#     logpdf-set(query-UNION-evidence) - logpdf-set(query)
# def test_factorization_of_joint_logpdf_set_in_two_columns():
#     """
#     p(row 1, row 0) = p(row 1| row 0) p(row 0)
#     log_p(row 1, row 0) = log_p(row 1| row 0) + log_p(row 0)
#     log_joint = log_conditional + log_marginal
#     """
#     view = simple_view
#     row1 = {1: {0: 0, 1: 0}}
#     row0 = {0: {0: 1, 1: 1}}

#     log_joint = view.logpdf_set(query=deep_merged(row0, row1))
#     log_conditional = view.logpdf_set(query=row1, evidence=row0)
#     log_marginal = view.logpdf_set(query=row0)

#     assert np.allclose(log_joint, log_conditional + log_marginal)

# ----- TEST MARGINALIZATION ----- #
# P(x) ~~ 1/N * sum_y P(x|y), with y ~ P(Y) 
# def test_marginalization_of_joint_logpdf_in_two_columns():
#     """
#     THIS MIGHT BE WRONG DUE TO THE DIFFERENCE BETWEEN HYPOTHETICAL AND NON-HYPOTHETICAL
#     p(row 1) = sum_{row 2} p(row 2, row 1)
#     log_p(row 1) = logsumexp_{row 2} log_p(row 2, row 1)
#     log_marginal = log_marginalized_joint
#     """
#     view = simple_view
#     row1 = {1: {0: 1, 1: 1}}
#     log_marginal = view.logpdf_set(query=row1)  # log_p(row1)
#     # p({1: {0: 1, 1: 1}}) = 1./4 * 1./2 + 4./9 * 1./2 

#     log_marginalized_joint = - np.float("inf") 
#     for values in product((0, 1), (0, 1)):  # marginalize values in row 2
#         row2 = {2: {c: i for c, i in zip([0, 1], values)}}
#         log_joint = view.logpdf_set(query=deep_merged(row2, row1))
#         log_marginalized_joint = logsumexp(
#             [log_marginalized_joint, log_joint])

#     assert np.allclose(log_marginal, log_marginalized_joint)

# def test_marginalization_of_joint_logpdf_in_one_column():
#     """
#     p({1: {0: 0}) = p({1: {0: 0}, 2: {0: 0}) + p({1: {0: 0}, 2: {0: 1})
#     """
#     view = simple_view
#     row1 = {1: {0: 0}}
#     log_marginal = view.logpdf_set(query=row1)  
#       # log_p(row1) = log(1./3 * 1./2 + 1./2 * 1./2)

#     log_marginalized_joint = - np.float("inf")  
#     for value in [0, 1]:  # marginalize values in row 2
#         row2 = {2: {0: value}}
#         log_joint = view.logpdf_set(query=deep_merged(row2, row1))
#         log_marginalized_joint = logsumexp(
#             [log_marginalized_joint, log_joint])

#     assert np.allclose(log_marginal, log_marginalized_joint)

# # ----- TEST BAYES RULE ----- #

# def test_bayes_inversion_of_logpdf_multirow_in_two_columns():
#     """
#     p(row 1 | row 2) = p(row 2| row 1) p(row 1) / p(row 2)
#     log_p(row 1 | row 2) = log_p(row 2| row 1) + log_p(row 1) - log_p(row 2)
#     log_posterior = log_likelihood + log_prior - log_marginal
#     """
#     view = simple_view
#     row1 = {1: {0: 0, 1: 0}}
#     row2 = {2: {0: 1, 1: 1}}

#     log_posterior = view.logpdf_set(query=row1, evidence=row2)

#     log_likelihood = view.logpdf_set(query=row2, evidence=row1)

#     log_prior = view.logpdf_set(query=row1)
#     assert np.allclose(log_prior, view.logpdf(rowid=1, query=row1[1]))

#     log_marginal = view.logpdf_set(query=row2)
#     # assert np.allclose(log_marginal, view.logpdf(rowid=0, query=row2[0]))

#     assert np.allclose(
#         log_posterior, log_likelihood + log_prior - log_marginal)

# def test_quick_query_logpdf_set_with_full_row():
#     view = simple_view

#     row0_short = {0: {}}
#     row0 = {0: {0: 1, 1: 1}}

#     assert np.allclose(
#         view.logpdf_set(row0_short), view.logpdf_set(row0))

#     with pytest.raises(ValueError):
#         view.logpdf_set({1: {}})

# def test_quick_query_logpdf_set_hypothetical():
#     view = simple_view
    
#     row1 = {1: {0: 1, 1: 1}}
#     row1_short = {-1: {0: 1, 1: 1}}

#     assert np.allclose(
#         view.logpdf_set(row1), view.logpdf_set(row1_short))

# def test_quick_query_joint_logpdf_set_hypothetical():
#     view = simple_view
    
#     row1 = {1: {0: 1, 1: 1}}
#     row2 = {2: {0: 1, 1: 1}}
#     row1_short = {-1: {0: 1, 1: 1}}
#     row2_short = {-2: {0: 1, 1: 1}}

#     assert np.allclose(
#         view.logpdf_set(deep_merged(row1, row2)),
#         view.logpdf_set(deep_merged(row1_short, row2_short)))

# def test_same_rowid_of_query_and_evidence_hypothetical():
#     view = simple_view
    
#     # P({1: {0: 1}} | {1: {0: 1}}) hypothetical row
#     row1 = {1: {0: 1}}
#     row2 = {2: {0: 1}}
#     answer = view.logpdf_set(query=row2, evidence=row1)
#     with pytest.warns(UserWarning):
#         test_out = view.logpdf_set(query=row1.copy(), evidence=row1)
#     assert np.allclose(answer, test_out)

# def test_same_rowid_of_query_and_evidence_nonhypothetical():
#     view = simple_view

#     # P({0: {0:1} | {0: {0: 1}) non-hypothetical row
#     row0 = {0: {0: 1}}
#     row1 = {1: {0: 1}}
#     answer = view.logpdf_set(query=row1, evidence=row0)
#     with pytest.warns(UserWarning):
#         test_out = view.logpdf_set(query=row0.copy(), evidence=row0)
#     assert np.allclose(answer, test_out)
