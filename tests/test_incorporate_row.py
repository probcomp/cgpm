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

import unittest

import numpy as np

from gpmcc import state
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


class IncorporateRowTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cctypes, cls.distargs = cu.parse_distargs(['normal','poisson',
            'bernoulli','lognormal','exponential','geometric','vonmises'])
        T, Zv, Zc = tu.gen_data_table(200, [1], [[.33, .33, .34]], cls.cctypes,
            cls.distargs, [.95]*len(cls.cctypes), rng=gu.gen_rng(0))
        cls.T = T.T
        cls.state = state.State(cls.T[:10,:], cls.cctypes,
            distargs=cls.distargs, rng=gu.gen_rng(0))
        cls.state.transition(N=5)

    def test_incorporate(self):
        # Incorporate row into cluster 0 for all views.
        previous = np.asarray([v.Nk[0] for v in self.state.views])
        self.state.incorporate_rows([self.T[10,:]], k=[[0]*len(self.state.views)])
        self.assertEqual([v.Nk[0] for v in self.state.views], list(previous+1))

        # Incorporate row into a singleton for all views.
        previous = np.asarray([len(v.Nk) for v in self.state.views])
        self.state.incorporate_rows([self.T[11,:]], k=[previous])
        self.assertEqual([len(v.Nk) for v in self.state.views], list(previous+1))

        # Unincorporate row from the singleton view just created.
        previous = np.asarray([len(v.Nk) for v in self.state.views])
        self.state.unincorporate_rows([11])
        self.assertEqual([len(v.Nk) for v in self.state.views], list(previous-1))

        # Undo the last step.
        previous = np.asarray([len(v.Nk) for v in self.state.views])
        self.state.incorporate_rows([self.T[11,:]], k=[previous])
        self.assertEqual([len(v.Nk) for v in self.state.views], list(previous+1))

        # Incorporate row without specifying a view.
        self.state.incorporate_rows([self.T[12,:]], k=None)

        # Incorporate row specifying different clusters.
        k = [None] * len(self.state.views)
        k[::2] = [1] * len(k[::2])
        previous = np.asarray([v.Nk[1] for v in self.state.views])
        self.state.incorporate_rows([self.T[13,:]], k=[k])
        for i in xrange(len(self.state.views)):
            if i%2 == 0:
                self.assertEqual(self.state.views[i].Nk[1], previous[i]+1)

        # Incorporate two rows with different clusterings.
        previous = np.asarray([v.Nk[0] for v in self.state.views])
        k = [0 for _ in xrange(len(self.state.views))]
        self.state.incorporate_rows(self.T[[14,15],:], k=[k,k])
        self.assertEqual(self.state.views[i].Nk[0], previous[i]+2)

        # Incoporate remaining rows in the default way.
        self.state.incorporate_rows(self.T[16:,:])
        self.assertEqual(self.state.n_rows(), len(self.T))

        # Unincorporate all rows except the last one using ascending.
        self.state.unincorporate_rows(xrange(1, len(self.T)))
        self.assertEqual(self.state.n_rows(), 1)

        # Reincorporate all rows.
        self.state.incorporate_rows(self.T[1:,:])
        self.assertEqual(self.state.n_rows(), len(self.T))

        # Unincorporate all rows except the last one using descending.
        self.state.unincorporate_rows(xrange(len(self.T)-1, 0, -1))
        self.assertEqual(self.state.n_rows(), 1)

        # Unincorporating last dim should raise.
        self.assertRaises(ValueError, self.state.unincorporate_rows, [0])

if __name__ == '__main__':
    unittest.main()
