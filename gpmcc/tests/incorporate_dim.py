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

class IncorporateDimTest(unittest.TestCase):

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
        T, Zv, Zc = tu.gen_data_table(n_rows, view_weights, cluster_weights,
            cls.cctypes, cls.distargs, separation)
        cls.T = T.T
        cls.state = state.State(cls.T[:,:2], cls.cctypes[:2], cls.distargs[:2],
            seed=0)
        cls.state.transition(N=10)

    def test_incorporate(self):
        # Incorporate a new dim into view[0].
        self.state.incorporate_dim(self.T[:,2], self.cctypes[2],
            self.distargs[2], v=0)
        self.assertEqual(self.state.Zv[2], 0)

        # Incorporate a new dim into a newly created singleton view.
        self.state.incorporate_dim(self.T[:,3], self.cctypes[3],
            self.distargs[3], v=len(self.state.Nv))
        self.assertEqual(self.state.Zv[3], len(self.state.Nv)-1)

        # Incorporate dim without specifying a view.
        self.state.incorporate_dim(self.T[:,4], self.cctypes[4],
            self.distargs[4])

        # Incorporate the rest of the dims in the default way.
        for i in xrange(5, len(self.cctypes)):
            self.state.incorporate_dim(self.T[:,i], self.cctypes[i],
                self.distargs[i])

if __name__ == '__main__':
    unittest.main()
