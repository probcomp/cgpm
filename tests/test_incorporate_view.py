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

import pytest
import numpy as np

from cgpm.mixtures.view import View

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

def test_incorporate_adds_to_dataset():
    view = initialize_view()

    rowid = 1
    row = {0: 1, 1: 1}
    view.incorporate(rowid=rowid, query=row)
    assert view.X[1][rowid] == 1

def test_incorporate_after_unincorporate_adds_to_dataset():
    view = initialize_view()

    rowid = 0
    row = {0: 1, 1: 1}
    view.unincorporate(rowid=0)
    view.incorporate(rowid=0, query=row)
    assert view.X[0][rowid] == 1

def test_error_when_incorporating_same_row_twice():
    view = initialize_view()

    row = {0: 1, 1: 1}
    view.incorporate(rowid=1, query=row)
    with pytest.raises(ValueError):
        view.incorporate(rowid=1, query=row)

def test_incorporate_samples_cluster_assignment_if_not_given():
    view = initialize_view()

    # Incorporate rows into successive clusters
    for k in xrange(1, 5):
        row = {0: 1, 1: 1, view.exposed_latent: k}
        view.incorporate(
            rowid=k, query=row)

    # Incorporate rows without cluster assigment
    test_lst = []
    for i in xrange(5, 15):
        row = {0: 1, 1: 1}
        view.incorporate(rowid=i, query=row)
        test_lst += [view.Zr()[i]]

    # Check that rows are not all zero 
    assert np.any(test_lst)

def test_incorporate_changes_cluster_assignment():
    view = initialize_view()

    row = {0: 1, 1: 1, 1000: 1}
    view.incorporate(rowid=1, query=row)
    assert view.Zr()[1] == 1
            
def test_incorporate_updates_bernoulli_suffstats():
    view = initialize_view()

    row = {0: 1, 1: 1, 1000: 0}
    view.incorporate(rowid=1, query=row)
    assert view.dims[0].clusters[0].x_sum == 2
    assert view.dims[0].clusters[0].N == 2
    assert view.dims[1].clusters[0].x_sum == 2
    assert view.dims[1].clusters[0].N == 2

def test_incorporate_with_missing_value_changes_cluster_assignment():
    view = initialize_view()

    row = {0: 1, 1000: 1}
    view.incorporate(rowid=1, query=row)
    assert view.Zr()[1] == 1

def test_incorporate_with_missing_value_updates_bernoulli_suffstats():
    view = initialize_view()

    row = {0: 1, 1000: 0}
    view.incorporate(rowid=1, query=row)
    assert view.dims[0].clusters[0].x_sum == 2
    assert view.dims[0].clusters[0].N == 2
    assert view.dims[1].clusters[0].x_sum == 1
    assert view.dims[1].clusters[0].N == 1
