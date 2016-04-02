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

from gpmcc.dists.forest import RandomForest

from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu

class RandomForestDirectTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n_rows = 50
        view_weights = [1]
        cluster_weights = np.array([[.33, .33, .34]])
        cls.cctypes = [
            'categorical(k=3)',
            'normal',
            'poisson',
            'bernoulli',
            'lognormal',
            'exponential',
            'geometric',
            'vonmises']
        separation = [.2] * len(cls.cctypes)
        cls.cctypes, cls.distargs = cu.parse_distargs(cls.cctypes)
        D, _, _ = tu.gen_data_table(n_rows, view_weights, cluster_weights,
            cls.cctypes, cls.distargs, separation, rng=gu.gen_rng(0))
        cls.D = D.T

    def test_incorporate(self):
        forest = RandomForest(
            distargs={'k':self.distargs[0]['k'], 'cctypes':self.cctypes[1:]})
        # Incorporate first 20 rows.
        for row in self.D[:20]:
            forest.incorporate(row[0], y=row[1:])
        # Unincorporating row 20 should raise.
        with self.assertRaises(ValueError):
            forest.unincorporate(self.D[20,0], y=self.D[20,1:])
        # Unincorporate all rows.
        for row in self.D[:20]:
            forest.unincorporate(row[0], y=row[1:])
        # Unincorporating row 0 should raise.
        with self.assertRaises(ValueError):
            forest.unincorporate(self.D[0,0], y=self.D[0,1:])
        # Incorporate some more rows.
        for row in self.D[:10]:
            forest.incorporate(row[0], y=row[1:])

    def test_logpdf_marginal(self):
        forest = RandomForest(
            distargs={'k':self.distargs[0]['k'], 'cctypes':self.cctypes[1:]})
        for row in self.D[:25]:
            forest.incorporate(row[0], y=row[1:])
        self.assertLess(forest.logpdf_marginal(), 0)

    def test_logpdf_predictive(self):
        forest = RandomForest(
            distargs={'k':self.distargs[0]['k'], 'cctypes':self.cctypes[1:]})
        Dx0 = self.D[self.D[:,0]==0]
        Dx1 = self.D[self.D[:,0]==1]
        Dx2 = self.D[self.D[:,0]==2]
        for row in Dx0[:-1]:
            forest.incorporate(row[0], y=row[1:])
        # Compute predictive for only seen class 0 which must be log(1)=0.
        self.assertEqual(forest.logpdf(Dx0[-1,0], y=Dx0[-1,1:]), 0)
        # Ensure can compute predictive for unseen classes, which will be.
        if len(Dx1) > 0:
            self.assertLess(forest.logpdf(Dx1[0,0], y=Dx1[0,1:]), 0)
        if len(Dx2) > 0:
            self.assertLess(forest.logpdf(Dx2[0,0], y=Dx2[0,1:]), 0)

    def test_simulate(self):
        forest = RandomForest(
            distargs={'k':self.distargs[0]['k'], 'cctypes':self.cctypes[1:]})
        for row in self.D[:25]:
            forest.incorporate(row[0], y=row[1:])
        correct, total = 0, 0.
        for row in self.D[25:]:
            s = forest.simulate(y=row[1:])
            correct += (s==row[0])
            total += 1.
        # Classification should be better than random.
        self.assertGreater(correct/total, 1./self.distargs[0]['k'])

if __name__ == '__main__':
    unittest.main()
