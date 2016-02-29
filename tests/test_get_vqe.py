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

import itertools
import unittest

import numpy as np

from gpmcc.state import State
from gpmcc.utils import validation as vu

class TestGetViewQueryEvidence(unittest.TestCase):

    def test_query_evidence(self):
        D = np.random.normal(size=(10,1))
        T = np.repeat(D, 10, axis=1)
        Zv = [0,0,0,1,1,1,2,2,2,3]

        state = State(T, ['normal']*10, Zv=Zv)

        query = [(9,1), (0,0), (1,1), (4,2), (5,7), (7,0)]
        evidence = [(3,1), (9,1), (6,-1)]
        queries, evidences = state._get_view_qe(query, evidence)

        # All 4 views have query.
        self.assertEqual(len(queries), 4)

        # View 0 has 2 queries.
        self.assertTrue(len(queries[0]), 2)
        self.assertIn((0,0), queries[0])
        self.assertIn((1,1), queries[0])
        # View 1 has 2 queries.
        self.assertTrue(len(queries[1]), 2)
        self.assertIn((4,2), queries[1])
        self.assertIn((5,7), queries[1])
        # View 2 has 1 queries.
        self.assertTrue(len(queries[2]), 1)
        self.assertIn((7,0), queries[2])
        # View 3 has 1 queries.
        self.assertTrue(len(queries[3]), 1)
        self.assertIn((9,1), queries[3])

        # Views 1,2,3 have evidence.
        self.assertEqual(len(evidences), 3)
        # View 1 has 2 evidence.
        self.assertTrue(len(evidences[1]), 2)
        self.assertIn((3,1), evidences[1])
        # View 2 has 1 evidence.
        self.assertTrue(len(evidences[2]), 1)
        self.assertIn((6,-1), evidences[2])
        # View 3 has 1 evidence.
        self.assertTrue(len(evidences[3]), 1)
        self.assertIn((9,1), evidences[3])

if __name__ == '__main__':
    unittest.main()
