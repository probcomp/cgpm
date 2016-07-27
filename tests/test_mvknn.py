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

import importlib

import json
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scipy.stats import chisquare
from scipy.stats import ks_2samp

from cgpm.knn.mvknn import MultivariateKnn
from cgpm.utils import general as gu
from cgpm.utils import test as tu


O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'categorical'


def test_initialize():
    # This test ensures that MvKnn raises on bad initialize arguments.
    # Typical initialization.
    MultivariateKnn(
        outputs=[0, 1], inputs=None, K=2,
        distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # No inputs allowed.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=[2], K=2,
            distargs={O: {ST:[N, C], SA: [{}, {'k': 2}]}})
    # At least one output.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[], inputs=[], K=2,
            distargs={O: {ST: [], SA:[]}})
    # Unique outputs.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 0], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure outputs in distargs.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs=None)
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={'output': {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure stattypes and statargs in distargs['outputs]'
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {'stattype': [N, C], SA :[{}, {'k': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], 'eland': [{}, {'k': 2}]}})
    # Ensure stattypes correct length.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C, N], SA: [{}, {'k': 2}]}})
    # Ensure statargs correct length.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, None, {'k': 2}]}})
    # Ensure number of categories provided as k.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, {'h': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=2,
            distargs={O: {ST: [N, C], SA: [{}, {}]}})
    # Missing number of nearest neighbors K.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Bad number of nearest neighbors K.
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=0,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKnn(
            outputs=[0, 1], inputs=None, K=-1,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})


def test_find_neighborhoods():
    # This test target _find_neighborhoods from MultivariateKnn. See the
    # inline comments for the description of each test.

    # Generate a high dimensional dataset with mixed numerical/categorical.
    rng = gu.gen_rng(1)

    outputs = range(11)
    inputs = None
    K = 5
    stattypes = [N, N, C, N, N, N, C, N, N, C, N]
    statargs = [{}, {}, {'k':7}, {}, {}, {}, {'k':1}, {}, {}, {'k':5}, {}]
    distargs = {'outputs': {'stattypes': stattypes, 'statargs': statargs}}

    X = rng.rand(100, 11)
    X[:,2] = rng.choice(range(statargs[2]['k']), size=100)
    X[:,6] = rng.choice(range(statargs[6]['k']), size=100)
    X[:,9] = rng.choice(range(statargs[9]['k']), size=100)
    X[:96,10] = np.nan

    knn = MultivariateKnn(outputs, inputs, K=K, distargs=distargs)

    for i, x in enumerate(X):
        knn.incorporate(i, dict(zip(outputs, x)))

    assert knn.N == len(X)

    # Neighbor search need evidence.
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence=None)
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence={})

    # Bad category 199.
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence={2:199, 5:.8})

    # Check the returned K and dimension are correct varying query/evidence.
    for q, e in [
            ([0,1], {5:.8}),
            ([0,1,7], {5:.8}),
            ([4], {5:.8, 7:0})
    ]:
        d, nh = knn._find_neighborhoods(query=q, evidence=e)
        assert len(nh) == K
        for n in nh:
            assert 1 <= len(n) <= K
            assert d[n].shape[1] == len(q)

    # Dimension 10 has only 4 non-nan values, so K=5 will fail.
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[0,1], evidence={10:.8})
    with pytest.raises(ValueError):
        knn._find_neighborhoods(query=[10,1], evidence={5:.8})

    # Now furnish dimension 10 with one additional non-nan value and ensure the
    # K=5 neighbors are the only possible ones i.e 0, 96, 97, 98, 99.
    knn.data[0][10] = 1

    def test_found_expected(dataset, neighborhoods, expect):
        for e in expect:
            match = False
            for n in neighborhoods:
                match = match or any(np.allclose(dataset[i], e) for i in n)
            assert match

    d_found, n_found = knn._find_neighborhoods(query=[0,1], evidence={10:.8})
    expected = [np.asarray(knn.data[r])[[0,1]]
        for r in [0, 96, 97, 98, 99]]
    test_found_expected(d_found, n_found, expected)

    d_found, n_found = knn._find_neighborhoods(query=[10,1], evidence={5:.8})
    expected = [np.asarray(knn.data[r])[[10,1]]
        for r in [0, 96, 97, 98, 99]]
    test_found_expected(d_found, n_found, expected)

    # Now make sure an exact match is in the nearest neighbor.
    z = knn.data[19]

    # # First crash since z contains a nan.
    with pytest.raises(ValueError):
        knn._find_neighborhoods([0,1], dict(zip(outputs[2:], z[2:])))

    # Now make sure that z is its own nearest neighbor.
    z_query = [0,1]
    z_evidence = {o:v for o,v in zip(outputs[2:], z[2:]) if not np.isnan(v)}
    d_found, n_found = knn._find_neighborhoods(z_query, z_evidence)
    test_found_expected(d_found, n_found, [z[:2]])
