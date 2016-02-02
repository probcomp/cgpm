# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
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

import unittest

import numpy as np

from gpmcc.utils import config as cu
from gpmcc.utils import test as tu
from gpmcc import state

class IncorporateRowTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n_rows = 200
        np.random.seed(0)
        view_weights = np.ones(1)
        cluster_weights = [np.array([.33, .33, .34])]
        cls.cctypes = [
            'normal',
            'poisson',
            'bernoulli',
            'lognormal',
            'exponential',
            'geometric',
            'vonmises']
        separation = [.95] * len(cls.cctypes)
        cls.cctypes, cls.distargs = cu.parse_distargs(cls.cctypes)
        T, _, _ = tu.gen_data_table(n_rows, view_weights, cluster_weights,
            cls.cctypes, cls.distargs, separation)
        cls.T = T.T
        cls.state = state.State(cls.T[:10,:], cls.cctypes, cls.distargs,
            seed=0)
        cls.state.transition(N=1)

    def test_incorporate(self):
        # Incorporate row into cluster 0 for all views.
        previous = np.asarray([v.Nk[0] for v in self.state.views])
        self.state.incorporate_row(self.T[10,:], k=[0]*len(self.state.views))
        self.assertEqual([v.Nk[0] for v in self.state.views], list(previous+1))

        # Incorporate row into a singleton for all views.
        previous = np.asarray([len(v.Nk) for v in self.state.views])
        self.state.incorporate_row(self.T[11,:], k=previous)
        self.assertEqual([len(v.Nk) for v in self.state.views], list(previous+1))

        # Unincorporate row from the singleton view just created.
        previous = np.asarray([len(v.Nk) for v in self.state.views])
        self.state.unincorporate_row(11)
        self.assertEqual([len(v.Nk) for v in self.state.views], list(previous-1))

        # Undo the last step.
        previous = np.asarray([len(v.Nk) for v in self.state.views])
        self.state.incorporate_row(self.T[11,:], k=previous)
        self.assertEqual([len(v.Nk) for v in self.state.views], list(previous+1))

        # Incorporate row without specifying a view.
        self.state.incorporate_row(self.T[12,:], k=None)

        # Incorporate row specifying different clusters.
        k = [None] * len(self.state.views)
        k[::2] = [1] * len(k[::2])
        previous = np.asarray([v.Nk[1] for v in self.state.views])
        self.state.incorporate_row(self.T[13,:], k=k)
        for i in xrange(len(self.state.views)):
            if i%2 == 0:
                self.assertEqual(self.state.views[i].Nk[1], previous[i]+1)

        # Incoporate all rows in the default way.
        for i in xrange(14, len(self.T)):
            self.state.incorporate_row(self.T[i,:])
        self.assertEqual(self.state.n_rows(), len(self.T))

        # Unincorporate all rows except the last one.
        for i in xrange(len(self.T)-1, 0, -1):
            self.state.unincorporate_row(i)
        self.assertEqual(self.state.n_rows(), 1)

        # Unincorporating last dim should raise.
        self.assertRaises(ValueError, self.state.unincorporate_row, 0)

if __name__ == '__main__':
    unittest.main()
