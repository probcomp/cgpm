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

import pytest
import numpy as np
from itertools import product

from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, deep_merged

# TODO: BUILD TESTS UPON EACH OTHER

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
        hypers={
            i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zr=[0])
    return view

def test_logpdf_multirow_in_singlerow_query_hypothetical():
    view = initialize_view()

    # LOGPDF
    # P(x[1,0] = 1) = 7./12
    # Hypothetical row: rowid=1
    query = {0: 1}
    math_out = np.log(7./12)
    test_out = view.logpdf(rowid=1, query=query)
    assert np.allclose(math_out, test_out)

    # LOGPDF MULTIROW
    # P(x[1,0] = 1) = 7./12
    # Hypothetical row: rowid=1
    query = {1: {0: 1}}
    math_out = np.log(7./12)
    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_logpdf_multirow_in_singlerow_query_nonhypothetical():
    view = initialize_view()

    # LOGPDF
    # P(x[0,0] = 1) = 1./2
    # Non-hypothetical row: rowid=0
    query = {0: 1}
    math_out = np.log(1./2)
    test_out = view.logpdf(rowid=0, query=query)
    # assert np.allclose(math_out, test_out)  # FAILING

    # LOGPDF MULTIROW
    # P(x[0,0] = 1) = 2./3
    # Non-hypothetical row: rowid=0
    query = {0: {0: 1}}
    math_out = np.log(1./2)
    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_logpdf_multirow_in_singlerow_cluster_hypothetical():
    view = initialize_view()

    # LOGPDF
    # P(z[1] = 0) = 1./2
    # Hypothetical row: rowid=1
    query = {view.exposed_latent: 0}
    math_out = np.log(1./2)
    test_out = view.logpdf(rowid=1, query=query)
    assert np.allclose(math_out, test_out)

    # LOGPDF MULTIROW
    # P(z[1] = 0) = 1./2
    # Hypothetical row: rowid=1
    query = {1: {view.exposed_latent: 0}}
    math_out = np.log(1./2)
    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)  # FAILS: math_out == 0

def test_logpdf_multirow_in_singlerow_cluster_nonhypothetical():
    view = initialize_view()

    # LOGPDF
    # P(z[0] = 0) = 1.
    # Non-hypothetical row: rowid=0
    query = {view.exposed_latent: 0}
    math_out = np.log(1)
    test_out = view.logpdf(rowid=0, query=query)
    assert np.allclose(math_out, test_out)

    # LOGPDF MULTIROW
    # P(z[0] = 0) = 1.
    # Non-hypothetical row: rowid=0
    query = {0: {view.exposed_latent: 0}}
    math_out = np.log(1)
    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_joint_logpdf_multirow_in_one_column():
    view = initialize_view()

    # P(x[0,0] = 1, x[1,0] = 1) = 7./24
    # (= 1./2 * 1 * (2./3 * 1./2 + 1./2 * 1./2)
    # Missing column and non-hypothetical row
    query = {0: {0: 1}, 1: {0: 1}}
    math_out = np.log(7./24)

    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_joint_logpdf_multirow_in_two_columns_with_missing_values():
    view = initialize_view()

    # P(x[0,0] = 1, x[1,1]=1)
    # = 1./2 * 1 * (1./2 * 1./2 + 1./2 * 1./2)
    query = {0: {0: 1}, 1: {1: 1}}
    math_out = np.log(1./4)

    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_joint_logpdf_multirow_in_one_column_with_cluster_assignments():
    view = initialize_view()

    # P(row 0: {0: 1, z: 0}, row 1: {0: 1, z: 1}) = 1./8
    # 1./2 * 1 * (1./2 * 1./2)
    z = view.exposed_latent
    query = {0: {0: 1, z: 0}, 1: {0: 1, z: 1}}
    math_out = np.log(1./8)

    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_joint_logpdf_multirow_in_one_column_conditioned_on_cluster_assignments():
    view = initialize_view()

    # p({row 0: {0: 1}, row 1: {0: 1}} | {0: {z: 0}, 1: {z: 1}}) = 1./4
    # = 1./8 * 1./(1./2)
    z = view.exposed_latent
    query = {0: {0: 1}, 1: {0: 1}}
    evidence = {0: {z: 0}, 1: {z: 1}}
    math_out = np.log(1./4)

    test_out = view.logpdf_multirow(query=query, evidence=evidence, debug=True)
    assert np.allclose(math_out, test_out)

def test_joint_logpdf_multirow_in_two_columns():
    view = initialize_view()

    # P(x[0,:] = [1,1], x[1,:] = [1,1]) = 25./288
    query = {0: {0: 1, 1: 1},
             1: {0: 1, 1: 1}}
    math_out = np.log(25./288)

    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

def test_logpdf_multirow_in_one_column_conditioned_on_another_row():
    view = initialize_view()

    # P(x[1,0] = 1 | x[0,0] = 1) = 7./12
    # Missing column and non-hypothetical row
    query = {1: {0: 1}}
    evidence = {0: {0: 1}}
    math_out = np.log(7./12)
    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

    # P(x[0,0] = 1 | x[1,0] = 1) = 7./12
    # Missing column and non-hypothetical row
    query = {0: {0: 1}}
    evidence = {1: {0: 1}}
    math_out = np.log(7./12)
    test_out = view.logpdf_multirow(query=query, evidence=evidence, debug=True)
    assert np.allclose(math_out, test_out)

def test_logpdf_multirow_in_two_columns_conditioned_on_another_row():
    view = initialize_view()

    # P(x[1,:] = [1,1] | x[0,:] = [1,1]) = 25./72
    query = {1: {0: 1, 1: 1}}
    evidence = {0: {0: 1, 1: 1}}
    math_out = np.log(25./72)
    test_out = view.logpdf_multirow(query=query, debug=True)
    assert np.allclose(math_out, test_out)

    # P(x[0,:] = [1,1] | x[1,:] = [1,1]) = 25./72
    query = {0: {0: 1, 1: 1}}
    evidence = {1: {0: 1, 1: 1}}
    math_out = np.log(25./72)
    test_out = view.logpdf_multirow(query=query, evidence=evidence, debug=True)
    assert np.allclose(math_out, test_out)

def test_factorization_of_joint_logpdf_multirow_in_two_columns():
    """
    p(row 1, row 0) = p(row 1| row 0) p(row 0)
    log_p(row 1, row 0) = log_p(row 1| row 0) + log_p(row 0)
    log_joint = log_conditional + log_marginal
    """
    view = initialize_view()
    row1 = {1: {0: 0, 1: 0}}
    row0 = {0: {0: 1, 1: 1}}

    log_joint = view.logpdf_multirow(query=deep_merged(row0, row1))
    log_conditional = view.logpdf_multirow(query=row1, evidence=row0)
    log_marginal = view.logpdf_multirow(query=row0)

    assert np.allclose(log_joint, log_conditional + log_marginal)

def test_marginalization_of_joint_logpdf_in_two_columns():
    """
    THIS MIGHT BE WRONG DUE TO THE DIFFERENCE BETWEEN HYPOTHETICAL AND NON-HYPOTHETICAL
    p(row 1) = sum_{row 2} p(row 2, row 1)
    log_p(row 1) = logsumexp_{row 2} log_p(row 2, row 1)
    log_marginal = log_marginalized_joint
    """
    view = initialize_view()
    row1 = {1: {0: 1, 1: 1}}
    log_marginal = view.logpdf_multirow(query=row1)  # log_p(row1)
    # p({1: {0: 1, 1: 1}}) = 1./4 * 1./2 + 4./9 * 1./2 

    log_marginalized_joint = - np.float("inf")  # initialize logsumexp to 0 in log space
    for values in product((0, 1), (0, 1)):  # marginalize values in row 2
        row2 = {2: {c: i for c, i in zip([0, 1], values)}}
        log_joint = view.logpdf_multirow(query=deep_merged(row2, row1))
        log_marginalized_joint = logsumexp(
            [log_marginalized_joint, log_joint])

    assert np.allclose(log_marginal, log_marginalized_joint)

def test_marginalization_of_joint_logpdf_in_one_column():
    """
    p({1: {0: 0}) = p({1: {0: 0}, 2: {0: 0}) + p({1: {0: 0}, 2: {0: 1})
    """
    view = initialize_view()
    row1 = {1: {0: 0}}
    log_marginal = view.logpdf_multirow(query=row1)  # log_p(row1) = log(1./3 * 1./2 + 1./2 * 1./2)

    log_marginalized_joint = - np.float("inf")  # initialize logsumexp to 0 in log space
    for value in [0, 1]:  # marginalize values in row 2
        row2 = {2: {0: value}}
        log_joint = view.logpdf_multirow(query=deep_merged(row2, row1))
        log_marginalized_joint = logsumexp(
            [log_marginalized_joint, log_joint])

    assert np.allclose(log_marginal, log_marginalized_joint)

def test_bayes_inversion_of_logpdf_multirow_in_two_columns():
    """
    p(row 1 | row 2) = p(row 2| row 1) p(row 1) / p(row 2)
    log_p(row 1 | row 2) = log_p(row 2| row 1) + log_p(row 1) - log_p(row 2)
    log_posterior = log_likelihood + log_prior - log_marginal
    """
    view = initialize_view()
    row1 = {1: {0: 0, 1: 0}}
    row2 = {2: {0: 1, 1: 1}}

    log_posterior = view.logpdf_multirow(query=row1, evidence=row2)

    log_likelihood = view.logpdf_multirow(query=row2, evidence=row1)

    log_prior = view.logpdf_multirow(query=row1)
    assert np.allclose(log_prior, view.logpdf(rowid=1, query=row1[1]))

    log_marginal = view.logpdf_multirow(query=row2)
    # assert np.allclose(log_marginal, view.logpdf(rowid=0, query=row2[0]))

    assert np.allclose(
        log_posterior, log_likelihood + log_prior - log_marginal)

def test_quick_query_logpdf_multirow_with_full_row():
    view = initialize_view()

    row0_short = {0 : {}}
    row0 = {0: {0: 1, 1: 1}}

    assert np.allclose(
        view.logpdf_multirow(row0_short), view.logpdf_multirow(row0))

    with pytest.raises(ValueError):
        view.logpdf_multirow({1: {}})

def test_quick_query_logpdf_multirow_hypothetical():
    view = initialize_view()
    
    row1 = {1: {0: 1, 1: 1}}
    row1_short = {-1: {0: 1, 1: 1}}

    assert np.allclose(
        view.logpdf_multirow(row1), view.logpdf_multirow(row1_short))


def test_same_rowid_of_query_and_evidence_hypothetical():
    view = initialize_view()
    
    # P({1: {0: 1}} | {1: {0: 1}}) hypothetical row
    row1 = {1: {0: 1}}
    row2 = {2: {0: 1}}
    math_out = view.logpdf_multirow(query=row2, evidence=row1)
    test_out = view.logpdf_multirow(query=row1, evidence=row1)
    assert np.allclose(math_out, test_out)

def test_same_rowid_of_query_and_evidence_nonhypothetical():
    view = initialize_view()

    # P({0: {0:1} | {0: {0: 1}) non-hypothetical row
    row0 = {0: {0: 1}}
    row1 = {1: {0:1 }}
    math_out = view.logpdf_multirow(query=row1, evidence=row0)
    test_out = view.logpdf_multirow(query=row0, evidence=row0)
    assert np.allclose(math_out, test_out)

 # TEST CONCORDANCE WITH LOGPDF
 # Deactivated, logpdf works differently for non-hypothetical rows
# def test_logpdf_in_test_joint_logpdf_factorizes():
#     view = initialize_view()
#     row0 = {0: {0: 1, 1: 1}}

#     log_marginal = view.logpdf_multirow(query=row0)
#     assert np.allclose(log_marginal, view.logpdf(rowid=0, query=row0[0]))

# def test_logpdf_in_test_joint_logpdf_marginalizes():
#     view = initialize_view()
#     row1 = {1: {0: 0, 1: 0}}
#     log_marginal = view.logpdf_multirow(query=row1)  # log_p(row1)
#     assert np.allclose(log_marginal, view.logpdf(rowid=1, query=row1[1]))

# def test_logpdf_in_test_bayes_inversion():
#     """
#     p(row 1 | row 2) = p(row 2| row 1) p(row 1) / p(row 2)
#     log_p(row 1 | row 2) = log_p(row 2| row 1) + log_p(row 1) - log_p(row 2)
#     log_posterior = log_likelihood + log_prior - log_marginal
#     """
#     view = initialize_view()
#     row1 = {1: {0: 0, 1: 0}}
#     row0 = {0: {0: 1, 1: 1}}

#     log_prior = view.logpdf_multirow(query=row1)
#     assert np.allclose(log_prior, view.logpdf(rowid=1, query=row1[1]))

#     log_marginal = view.logpdf_multirow(query=row0)
#     assert np.allclose(log_marginal, view.logpdf(rowid=0, query=row0[0]))

