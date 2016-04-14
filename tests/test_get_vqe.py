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

import itertools
import unittest

import numpy as np

from gpmcc.state import State
from gpmcc.utils import general as gu
from gpmcc.utils import validation as vu

class TestGetViewQueryEvidence(unittest.TestCase):

    def test_query_evidence(self):
        D = np.random.normal(size=(10,1))
        T = np.repeat(D, 10, axis=1)
        Zv = [0,0,0,1,1,1,2,2,2,3]

        state = State(T, ['normal']*10, Zv=Zv, rng=gu.gen_rng(0))

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
