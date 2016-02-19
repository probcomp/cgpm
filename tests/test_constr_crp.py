# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Checks the validate_crp_constrained_input functions catches bad cases."""

import unittest

from gpmcc.utils import validation as vu

class TestValidateCrpConsrainedInput(unittest.TestCase):

    def test_duplicate_dependence(self):
        Cd = [[0,2,3], [4,5,0]]
        with self.assertRaises(ValueError):
            vu.validate_crp_constrained_input(6, Cd ,[])

    def test_single_column_dependence(self):
        Cd = [[0], [4,5,2]]
        with self.assertRaises(ValueError):
            vu.validate_crp_constrained_input(6, Cd, [])

    def test_contradictory_independece(self):
        Cd = [[0,1,3], [2,4]]
        Ci = [(0,1)]
        with self.assertRaises(ValueError):
            vu.validate_crp_constrained_input(5, Cd, Ci)

    def test_valid_constraints(self):
        Cd = [[0,3], [2,4], [5,6]]
        Ci = [(0,2), (5,2)]
        vu.validate_crp_constrained_input(7, Cd, Ci)

if __name__ == '__main__':
    unittest.main()
