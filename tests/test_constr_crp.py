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
import unittest

from gpmcc.utils import validation as vu
from gpmcc.utils import general as gu

class TestValidateCrpConsrainedInput(unittest.TestCase):

    def test_duplicate_dependence(self):
        Cd = [[0,2,3], [4,5,0]]
        Ci = []
        Rd = Ri = {}
        with self.assertRaises(ValueError):
            vu.validate_crp_constrained_input(6, Cd , Ci, Rd, Ri)

    def test_single_customer_dependence(self):
        Cd = [[0], [4,5,2]]
        Ci = []
        Rd = Ri = {}
        with self.assertRaises(ValueError):
            vu.validate_crp_constrained_input(6, Cd, Ci, Rd, Ri)

    def test_contradictory_independece(self):
        Cd = [[0,1,3], [2,4]]
        Ci = [(0,1)]
        Rd = Ri = {}
        with self.assertRaises(ValueError):
            vu.validate_crp_constrained_input(5, Cd, Ci, Rd, Ri)

    def test_valid_constraints(self):
        Cd = [[0,3], [2,4], [5,6]]
        Ci = [(0,2), (5,2)]
        Rd = Ri = {}
        self.assertTrue(vu.validate_crp_constrained_input(7, Cd, Ci, Rd, Ri))

class TestSimulateCrpConstrained(unittest.TestCase):

    def test_no_constraints(self):
        N, alpha = 10, .4
        Cd = Ci = []
        Rd = Ri = {}
        Z = gu.simulate_crp_constrained(N, alpha, Cd, Ci, Rd, Ri)
        self.assertTrue(
            vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri))

    def test_all_friends(self):
        N, alpha = 10, 1.4
        Cd = [range(N)]
        Ci = []
        Rd = Ri = {}
        Z = gu.simulate_crp_constrained(N, alpha, Cd, Ci, Rd, Ri)
        self.assertTrue(
            vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri))

    def test_all_enemies(self):
        N, alpha = 13, 1.4
        Cd = []
        Ci = list(itertools.combinations(range(N), 2))
        Rd = Ri = {}
        Z = gu.simulate_crp_constrained(N, alpha, Cd, Ci, Rd, Ri)
        self.assertTrue(
            vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri))

    def test_complex_relationships(self):
        N, alpha = 15, 10
        Cd = [(0,1,4), (2,3,5), (8,7)]
        Ci = [(2,8), (0,3)]
        Rd = Ri = {}
        Z = gu.simulate_crp_constrained(N, alpha, Cd, Ci, Rd, Ri)
        self.assertTrue(
            vu.validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri))

if __name__ == '__main__':
    unittest.main()
