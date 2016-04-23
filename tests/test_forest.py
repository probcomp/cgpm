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

from math import log

import numpy as np

from scipy.misc import logsumexp

from gpmcc.dim import Dim
from gpmcc.dists.forest import RandomForest
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


class RandomForestDirectTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cctypes, distargs = cu.parse_distargs(['categorical(k=3)',
            'normal','poisson','bernoulli','lognormal','exponential',
            'geometric','vonmises'])
        D, Zv, Zc = tu.gen_data_table(50, [1], [[.33, .33, .34]], cctypes,
            distargs, [.2]*len(cctypes), rng=gu.gen_rng(0))
        cls.D = D.T
        cls.rf_distargs = {'k':distargs[0]['k'], 'cctypes':cctypes[1:]}
        cls.num_classes = 3

    def test_incorporate(self):
        forest = RandomForest(distargs=self.rf_distargs, rng=gu.gen_rng(0))
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

    def test_logpdf_uniform(self):
        """No observations implies uniform."""
        forest = RandomForest(distargs=self.rf_distargs, rng=gu.gen_rng(0))
        forest.transition_params()
        for x in xrange(self.num_classes):
            self.assertAlmostEqual(
                forest.logpdf(x, y=self.D[0,1:]), -log(self.num_classes))

    def test_logpdf_normalized(self):
        def train_on(c):
            D = [row for row in self.D if row[0] in c]
            forest = RandomForest(distargs=self.rf_distargs, rng=gu.gen_rng(0))
            for row in D:
                forest.incorporate(row[0], y=row[1:])
            forest.transition_params()
            return forest

        def test_on(forest, c):
            D = [row for row in self.D if row[0] not in c]
            for row in D:
                lps = [forest.logpdf(x, y=row[1:]) for x in
                    xrange(self.num_classes)]
                self.assertAlmostEqual(logsumexp(lps), 0)

        forest = train_on([])
        test_on(forest, [])

        forest = train_on([2])
        test_on(forest, [2])

        forest = train_on([0,1])
        test_on(forest, [0,1])

    def test_logpdf_marginal(self):
        forest = RandomForest(distargs=self.rf_distargs, rng=gu.gen_rng(0))
        for row in self.D[:25]:
            forest.incorporate(row[0], y=row[1:])
        forest.transition_params()
        self.assertLess(forest.logpdf_marginal(), 0)

    def test_transition_hypers(self):
        forest = Dim(
            'random_forest', 0, distargs=self.rf_distargs, rng=gu.gen_rng(0))
        forest.transition_hyper_grids(self.D[:,0])
        # Create two clusters.
        Zr = np.zeros(len(self.D), dtype=int)
        Zr[len(self.D)/2:] = 1
        forest.bulk_incorporate(self.D[:,0], Zr, Y=self.D[:,1:])
        # Transitions.
        forest.transition_params()
        forest.transition_hypers()

    def test_simulate(self):
        forest = Dim(
            'random_forest', 0, distargs=self.rf_distargs, rng=gu.gen_rng(0))
        forest.transition_hyper_grids(self.D[:,0])
        # Create 1 clusters.
        Zr = np.zeros(len(self.D[:40]), dtype=int)
        forest.bulk_incorporate(self.D[:40,0], Zr, Y=self.D[:40,1:])
        # Transitions.
        forest.transition_params()
        for _ in xrange(2):
            forest.transition_hypers()
        correct, total = 0, 0.
        for row in self.D[40:]:
            s = [forest.simulate(0, y=row[1:]) for _ in xrange(10)]
            s = np.argmax(np.bincount(s))
            correct += (s==row[0])
            total += 1.
        # Classification should be better than random.
        self.assertGreater(correct/total, 1./self.num_classes)

if __name__ == '__main__':
    unittest.main()
