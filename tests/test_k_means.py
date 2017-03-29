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

import k_means

def test_crash_simulate():
    fm = k_means.KMeans([0,1], [], K=2)
    fm.simulate(None, [0,1], N=2)

def test_crash_logpdf():
    raise NotImplementedError

def test_crash_transition():
    raise NotImplementedError

# Test whether incorp. and unincorp do the right thing.

def test_incorporate():
    """Test that after we incorporate for a certain rowid, the dict fm.data has
    a corresponding entry."""
    fm = k_means.KMeans([0,1], [], K=2)
    fm.incorporate(0, {0:2, 1:3})
    assert fm.data == {0:[2,3]}

def test_unincorporate():
    """Test that after we unincorporate for a certain rowid, the dict fm.data
    has a corresponding entry.
    """
    fm = k_means.KMeans([0,1], [], K=2)
    fm.incorporate(0, {0:2, 1:3})
    fm.unincorporate(0)

# Test whether simulate does the right thing.
def test_compare_to_single_Gaussian():
    """ Test whether the distubiton of K-means with K>2 is significantly
    different from a single Gaussian.
    """
    raise NotImplementedError

def test_compare_to_single_Gaussian():
    """ Test with a heuristic method that the data generated really implies that
    the number of clusters indeed equals K.

    Synthetically create data that clearly clusters visisbly. Adjust distance of
    clusters and distance of points inside a cluster.

    E.G using an elbo method  or a AIC or BIC criterion.
    """
    raise NotImplementedError

# Test transition.
def test_K_inferred_means_compared_with_K_known_means():
    raise NotImplementedError

def test_K_inferred_variances_compared_with_K_known_variances():
    raise NotImplementedError

# Test whether logpdf given generated data does the right thing.
def test_logdf_compare_k_equals_1_normal():
    """Compare the output of logpdf for a k-means CGPM with K=1 with the logdpf
    for a normal disribution."""
    raise NotImplementedError

