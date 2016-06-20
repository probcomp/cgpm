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

"""Checks the simulate_crp_constrained produces valid partitions."""

import itertools
import pytest

from cgpm.utils import general as gu
from cgpm.utils import validation as vu


# Tests for validate_crp_constrained.


def test_duplicate_Cd():
    Cd = [[0,2,3], [4,5,0]]
    Ci = []
    Rd = Ri = {}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(6, Cd , Ci, Rd, Ri)

def test_duplicate_dependence_Rd():
    Cd = [[2,3], [4,5,0]]
    Ci = []
    Rd = {2: [[0,1], [1,1]]}
    Ri = {}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(6, Cd , Ci, Rd, Ri)

def test_single_customer_dependence_Cd():
    Cd = [[0], [4,5,2]]
    Ci = []
    Rd = Ri = {}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(6, Cd, Ci, Rd, Ri)

def test_single_customer_dependence_Ri():
    Cd = [[0,1], [4,5,2]]
    Ci = []
    Rd = {0: [[1,3,4], [10]]}
    Ri = {}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(6, Cd, Ci, Rd, Ri)

def test_contradictory_independece_Cdi():
    Cd = [[0,1,3], [2,4]]
    Ci = [(0,1)]
    Rd = Ri = {}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(5, Cd, Ci, Rd, Ri)

def test_contradictory_independece_Rdi():
    Cd = [[0,1,3], [2,4]]
    Ci = [(1,4)]
    Rd = {2: [[0,1,5]]}
    Ri = {2: [(0,5)]}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(5, Cd, Ci, Rd, Ri)

def test_contradictory_independece_Cdi_Rdi():
    Cd = [[0,1,3], [2,4]]
    Ci = [(1,4)]
    Rd = {1: [[0,1,5], [6,2]]}
    Ri = {3: [(0,1)]}
    with pytest.raises(ValueError):
        vu.validate_crp_constrained_input(5, Cd, Ci, Rd, Ri)

def test_valid_constraints():
    Cd = [[0,3], [2,4], [5,6]]
    Ci = [(0,2), (5,2)]
    Rd = {0:[[1,4]], 3:[[1,2], [9,5]]}
    Ri = {0:[(1,19)], 4:[(1,2)]}
    assert vu.validate_crp_constrained_input(7, Cd, Ci, Rd, Ri)


# Tests for simulate_crp_constrained.


def test_no_constraints():
    N, alpha = 10, .4
    Cd = Ci = []
    Rd = Ri = {}
    Z = gu.simulate_crp_constrained(
        N, alpha, Cd, Ci, Rd, Ri, rng=gu.gen_rng(0))
    assert vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)

def test_all_friends():
    N, alpha = 10, 1.4
    Cd = [range(N)]
    Ci = []
    Rd = Ri = {}
    Z = gu.simulate_crp_constrained(
        N, alpha, Cd, Ci, Rd, Ri, rng=gu.gen_rng(0))
    assert vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)

def test_all_enemies():
    N, alpha = 13, 1.4
    Cd = []
    Ci = list(itertools.combinations(range(N), 2))
    Rd = Ri = {}
    Z = gu.simulate_crp_constrained(
        N, alpha, Cd, Ci, Rd, Ri, rng=gu.gen_rng(0))
    assert vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)

def test_all_enemies_rows():
    # The row constraints will force all columns to be independent.
    N, alpha = 3, 1
    Cd = []
    Ci = []
    Rd = {0:[[0,1]], 1:[[1,2]], 2:[[2,3]]}
    Ri = {0:[(1,2)], 1:[(2,3)], 2:[(0,1)]}
    Z = gu.simulate_crp_constrained(
        N, alpha, Cd, Ci, Rd, Ri, rng=gu.gen_rng(0))
    assert vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)

def test_complex_relationships():
    N, alpha = 15, 10
    Cd = [(0,1,4), (2,3,5), (8,7)]
    Ci = [(2,8), (0,3)]
    Rd = Ri = {}
    Z = gu.simulate_crp_constrained(
        N, alpha, Cd, Ci, Rd, Ri, rng=gu.gen_rng(0))
    assert vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)
