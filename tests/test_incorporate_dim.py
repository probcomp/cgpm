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

class IncorporateDimTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cctypes, cls.distargs = cu.parse_distargs(['normal','poisson',
            'bernoulli','lognormal','exponential','geometric','vonmises'])
        T, Zv, Zc = tu.gen_data_table(200, [1], [[.33, .33, .34]], cls.cctypes,
            cls.distargs, [.95]*len(cls.cctypes), rng=gu.gen_rng(0))
        cls.T = T.T
        cls.state = state.State(cls.T[:,:2], cls.cctypes[:2],
            distargs=cls.distargs[:2], rng=gu.gen_rng(0))
        cls.state.transition(N=5)

    def test_incorporate(self):
        # Incorporate a new dim into view[0].
        self.state.incorporate_dim(self.T[:,2], self.cctypes[2],
            self.distargs[2], v=0)
        self.assertEqual(self.state.Zv[2], 0)

        # Incorporate a new dim into a newly created singleton view.
        self.state.incorporate_dim(self.T[:,3], self.cctypes[3],
            self.distargs[3], v=len(self.state.views))
        self.assertEqual(self.state.Zv[3], len(self.state.views)-1)

        # Incorporate dim without specifying a view.
        self.state.incorporate_dim(self.T[:,4], self.cctypes[4],
            self.distargs[4])

        # Unincorporate first dim.
        previous = len(self.state.Zv)
        self.state.unincorporate_dim(0)
        self.assertEqual(len(self.state.Zv), previous-1)

        # Reincorporate dim without specifying a view.
        self.state.incorporate_dim(self.T[:,0], self.cctypes[0],
            self.distargs[0])

        # Incorporate dim into singleton view, remove it, assert destroyed.
        self.state.incorporate_dim(self.T[:,5], self.cctypes[5],
            self.distargs[5], v=len(self.state.views))
        previous = len(self.state.views)
        self.state.unincorporate_dim(5)
        self.assertEqual(len(self.state.views), previous-1)

        # Reincorporate dim into a singleton view.
        self.state.incorporate_dim(self.T[:,5], self.cctypes[4],
            self.distargs[4], v=len(self.state.views))

        # Incorporate the rest of the dims in the default way.
        for i in xrange(6, len(self.cctypes)):
            self.state.incorporate_dim(self.T[:,i], self.cctypes[i],
                self.distargs[i])
        self.assertEqual(self.state.n_cols(), self.T.shape[1])

        # Unincorporate all the dims, except the last one.
        for i in xrange(self.state.n_cols()-1, 0, -1):
            self.state.unincorporate_dim(i)
        self.assertEqual(self.state.n_cols(), 1)

        # Unincorporating last dim should raise.
        self.assertRaises(ValueError, self.state.unincorporate_dim, 0)

if __name__ == '__main__':
    unittest.main()
