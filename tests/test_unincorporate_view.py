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

def test_unincorporate_removes_from_dataset():
    view = initialize_view()

    row = {0: 1, 1: 1}
    view.unincorporate(0)
    assert not view.X[0]

def test_error_when_unincorporating_same_row_twice():
    view = initialize_view()

    row = {0: 1, 1: 1}
    view.incorporate(rowid=2, query=row)
    view.unincorporate(rowid=2)
    with pytest.raises(ValueError):
        view.unincorporate(rowid=1)

def test_unincorporate_changes_row_cluster_assignment():
    view = initialize_view()

    row = {0: 1, 1: 1, 1000: 1}
    view.incorporate(rowid=1, query=row)
    view.unincorporate(rowid=1)
    assert 1 not in view.Zr().keys()

def test_unincorporate_updates_bernoulli_suffstats():
    view = initialize_view()

    row = {0: 1, 1: 1, 1000: 0}
    view.incorporate(rowid=1, query=row)
    view.unincorporate(rowid=1)
    assert view.dims[0].clusters[0].x_sum == 1
    assert view.dims[0].clusters[0].N == 1
    assert view.dims[1].clusters[0].x_sum == 1
    assert view.dims[1].clusters[0].N == 1

def test_unincorporate_with_missing_value_changes_row_assignment():
    view = initialize_view()

    row = {0: 1, 1000: 1}
    view.incorporate(rowid=1, query=row)
    view.unincorporate(rowid=1)
    assert 1 not in view.Zr().keys()

def test_unincorporate_with_missing_value_changes_bernoulli_suffstats():
    view = initialize_view()

    row = {0: 1, 1000: 0}
    view.incorporate(rowid=1, query=row)
    view.unincorporate(rowid=1)
    assert view.dims[0].clusters[0].x_sum == 1
    assert view.dims[0].clusters[0].N == 1
    assert view.dims[1].clusters[0].x_sum == 1
    assert view.dims[1].clusters[0].N == 1
