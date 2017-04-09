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


# Crash for the basic functions (simulate, logpdf).
import sys
#TODO: create setup.py
sys.path.insert(0, 'src/k_means')

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import kstest
from scipy.stats import ks_2samp

import k_means
from stochastic import stochastic

# Test whether incorp. and unincorp do the right thing.

def test_incorporate():
    """Test that after we incorporate for a certain rowid, the dict km.data has
    a corresponding entry."""
    km = k_means.KMeans([0,1], [], K=2)
    km.incorporate(0, {0:2, 1:3})
    assert km.data == {0:[2,3]}

def test_unincorporate():
    """Test that after we unincorporate for a certain rowid, the dict km.data
    has a corresponding entry.
    """
    km = k_means.KMeans([0,1], [], K=2)
    km.incorporate(0, {0:2, 1:3})
    km.unincorporate(0)

# Test whether simulate does the right thing.
def test_simulate_crash():
    km = k_means.KMeans([0,1], [], K=2)
    query = [0]
    rowid = None
    samples = km.simulate(rowid, query, N=100)

@stochastic(max_runs=3, min_passes=1)
def test_simulate_compare_to_single_Gaussian(seed):
    """ Test whether the distubiton of K-means with K=1 is significantly
    different from a single Gaussian.
    """
    km = k_means.KMeans([0,1], [], K=2)
    column = 0
    query = [column]
    rowid = None
    samples = km.simulate(rowid, query, N=100)
    ks_test_result = kstest([sample[column] for sample in samples], 'norm')
    assert ks_test_result.pvalue > 0.05

@stochastic(max_runs=3, min_passes=1)
def test_simulate_dim_marginals(seed):
    """Test that the distribution is correct, by using a KS two sample test on
    the individual dimensions.
    """
    km = k_means.KMeans(
        [0,1],
        [],
        K=2,
        params={'cluster_centers':[np.array([-1, -1]), np.array([2,2]),]}
    )
    query = [0,1]
    rowid = None
    samples = km.simulate(rowid, query, N=1000)

    # compare by just sampling from two known clusters.
    samples_k1 = np.random.multivariate_normal([-1,-1], [[1,0],[0,1]], 500)
    samples_k2 = np.random.multivariate_normal([2,2], [[1,0],[0,1]], 500)
    marginal_expected_dim1 = np.concatenate((samples_k1[:,0], samples_k2[:,0]))
    marginal_expected_dim2 = np.concatenate((samples_k1[:,1], samples_k2[:,1]))
    marginal_actual_dim1 = [sample[0] for sample in samples]
    marginal_actual_dim2 = [sample[1] for sample in samples]

    ks_test_result_dim1 = ks_2samp(marginal_expected_dim1, marginal_actual_dim1)
    ks_test_result_dim2 = ks_2samp(marginal_expected_dim2, marginal_actual_dim2)

    assert ks_test_result_dim1.pvalue > 0.05
    assert ks_test_result_dim2.pvalue > 0.05

# helper, for bulk incorporate
def bulk_incorporate(data_table, cgpm_object):
    for i,row in enumerate(data_table):
        cgpm_object.incorporate(
            i,
            {j:value for j,value in enumerate(row)}
        )

def test_transition_crash():
    km = k_means.KMeans([0,1], [], K=2)
    km.incorporate(0, {0:2, 1:3})
    km.incorporate(1, {0:1, 1:4})
    query = [0]
    rowid = None
    km.transition()

def test_transition_K_inferred_distribution():
    """Similar to test dim_marginals, but actually infererring the distribution.
    """
    km = k_means.KMeans(
        [0,1],
        [],
        K=2,
    )
    training_samples_k1 = np.random.multivariate_normal(
        [-1,-1], [[1,0],[0,1]], 100
    )
    training_samples_k2 = np.random.multivariate_normal(
        [2,2], [[1,0],[0,1]], 100
    )
    bulk_incorporate(
        np.concatenate((training_samples_k1, training_samples_k2)),
        km
    )
    km.transition()
    query = [0,1]
    rowid = None
    samples = km.simulate(rowid, query, N=200)

    # compare by just sampling from two known clusters.
    samples_k1 = np.random.multivariate_normal([-1,-1], [[1,0],[0,1]], 100)
    samples_k2 = np.random.multivariate_normal([2,2], [[1,0],[0,1]], 100)
    marginal_expected_dim1 = np.concatenate((samples_k1[:,0], samples_k2[:,0]))
    marginal_expected_dim2 = np.concatenate((samples_k1[:,1], samples_k2[:,1]))
    marginal_actual_dim1 = [sample[0] for sample in samples]
    marginal_actual_dim2 = [sample[1] for sample in samples]

    ks_test_result_dim1 = ks_2samp(marginal_expected_dim1, marginal_actual_dim1)
    ks_test_result_dim2 = ks_2samp(marginal_expected_dim2, marginal_actual_dim2)

    assert ks_test_result_dim1.pvalue > 0.05
    assert ks_test_result_dim2.pvalue > 0.05

@stochastic(max_runs=3, min_passes=1)
def test_transition_K_inferred_means_compared_with_K_known_means(seed):
    """Testing whether the inferred means are roughly correct.
    """
    km = k_means.KMeans(
        [0,1],
        [],
        K=2,
    )
    cov =  [[1,0],[0,1]]
    mean_1 = [-1,-1]
    mean_2 = [2, 2]
    training_samples_k1 = np.random.multivariate_normal(mean_1, cov, 200)
    training_samples_k2 = np.random.multivariate_normal(mean_2, cov, 200)
    bulk_incorporate(
        np.concatenate((training_samples_k1, training_samples_k2)),
        km
    )
    km.transition()
    # Accounting for the fact that cluster centers are interchangebable.
    centers = sorted(km.cluster_centers)
    np.testing.assert_almost_equal(centers[0], mean_1, decimal=1)
    np.testing.assert_almost_equal(centers[1], mean_2, decimal=1)

@stochastic(max_runs=5, min_passes=1)
def test_transition_K_inferred_variances_compared_with_K_known_variances(seed):
    """Testing whether the inferred sigmas are roughly correct.
    """
    km = k_means.KMeans(
        [0,1],
        [],
        K=2,
    )
    mean_1 = [-5,-5]
    mean_2 = [2, 2]
    sigma_1 = 3
    sigma_2 = 0.1
    cov_1 =  [[sigma_1, 0],[0, sigma_1]]
    cov_2 =  [[sigma_2, 0],[0, sigma_2]]
    training_samples_k1 = np.random.multivariate_normal(mean_1, cov_1, 500)
    training_samples_k2 = np.random.multivariate_normal(mean_2, cov_2, 500)
    bulk_incorporate(
        np.concatenate((training_samples_k1, training_samples_k2)),
        km
    )
    km.transition()
    # Accounting for the fact that cluster centers are interchangebable.
    sigmas = sorted(km.cluster_sigmas, reverse=True)
    np.testing.assert_almost_equal(sigmas[0], sigma_1, decimal=1)
    np.testing.assert_almost_equal(sigmas[1], sigma_2, decimal=1)

# Test whether logpdf given generated data does the right thing.
def test_logpdf_crash():
    """Test whether a call to logpdf crashes."""
    km = k_means.KMeans([0,1], [], K=1)
    rowid = None
    query = {0:2.4, 1:1.2}
    km.logpdf(rowid, query)

def test_logpdf_compare_k_equals_1_normal():
    """Compare the output of logpdf for a k-means CGPM with K=1 with the logdpf
    for a normal disribution."""
    km = k_means.KMeans([0,1], [], K=1)
    rowid = None
    query = {0:2.4, 1:1.2}
    scipy_mvn = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
    assert scipy_mvn.pdf([2.4, 1.2]) == np.exp(km.logpdf(rowid, query))

def test_logpdf_compare_k_equals_2_mixture():
    """Compare the output of logpdf for a k-means CGPM with K=2 with the logdpf
    for a normal disribution."""
    km = k_means.KMeans(
        [0,1],
        [],
        K=2,
        params={'cluster_centers':[np.array([1,1]), np.array([2,2]),]}
    )
    rowid = None
    query = {0:2.4, 1:1.2}
    scipy_mvn_k1 = multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]])
    scipy_mvn_k2 = multivariate_normal(mean=[2,2], cov=[[1,0],[0,1]])
    # Test assumes that mixing coefficients are both 0.5.
    expected =\
        np.log((scipy_mvn_k1.pdf([2.4, 1.2]) + scipy_mvn_k2.pdf([2.4, 1.2]))/2.)
    actual = km.logpdf(rowid, query)
    assert expected == actual

def test_logpdf_compare_k_equals_1_normal_with_evidence():
    """Compare the output of conditioned logpdf for a k-means CGPM with K=1 with
    the logdpf for a normal disribution.

    Since we restrict the MV to have covariance sigma * I, the joint
    multivariate normal for col1 and col2 can be written as
        p(col1, col2) =  p(col1) * p(col1)
    Thus, co1 and col2 are independent. Ergo, p(col1 | col2)  = p(col1).
    """
    km = k_means.KMeans([0,1], [], K=1)
    rowid = None
    query = {0:2.4}
    evidence = {1:1.2}
    scipy_mvn = multivariate_normal(mean=[0], cov=[1])
    assert scipy_mvn.pdf([2.4]) ==\
        np.exp(km.logpdf(rowid, query, evidence=evidence))

def test_logpdf_compare_k_equals_2_mixture_conditional():
    """Compare the output of logpdf for a k-means CGPM with K=2 with the logdpf
    for a normal disribution. Same as above. This time however, we supply
    evidence."""
    km = k_means.KMeans(
        [0,1],
        [],
        K=2,
        params={'cluster_centers':[np.array([1,1]), np.array([2,2]),]}
    )
    rowid = None
    query = {0:2.4}
    evidence = {1:1.2}
    # XXX is this correct?
    scipy_mvn_k1 = multivariate_normal(mean=[1], cov=[1])
    scipy_mvn_k2 = multivariate_normal(mean=[2], cov=[1])
    # Test assumes that mixing coefficients are both 0.5.
    W_1 = 0.5 * scipy_mvn_k1.pdf([1.2])
    W_2 = 0.5 * scipy_mvn_k2.pdf([1.2])
    W_denominator = W_1 + W_2
    W_1 = W_1/W_denominator
    W_2 = W_2/W_denominator
    expected = np.log(W_1 * scipy_mvn_k1.pdf([2.4]) + W_2 * scipy_mvn_k2.pdf([2.4]))
    actual = km.logpdf(rowid, query, evidence=evidence)
    assert expected == actual
