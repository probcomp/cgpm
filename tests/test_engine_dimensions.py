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

import unittest

import numpy as np

from gpmcc.utils import config as cu
from gpmcc.utils import test as tu
from gpmcc import engine

"""This test suite ensures that results returned from Engine.logpdf,
Engine.logpdf_bulk, Engine.simulate, and Engine.simulate_bulk are of the correct
dimensions, based on the number of states and type of query.

Every results from these queries should be a list of length Engine.num_states.
The elemnts of the returned list differ based on the method, where we use

    - logpdf[s] = logpdf of query from state s.

    - simulate[s][i] = sample i (i=1..N) of query from state s.

    - logpdf_bulk[s][j] = logpdf of query[j] from state s.

    - simulate_bulk[s][j][i] = sample (i=1..Ns[j]) of query[j] from state s.

This test suite is slow because many simulate/logpdf queries are invoked.
"""

class EngineDimensionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up the data generation
        n_rows = 100
        view_weights = np.ones(1)
        cluster_weights = [[.25, .25, .5]]
        cctypes = [
            'normal',
            'poisson',
            'bernoulli',
            'lognormal',
            'beta_uc',
            'vonmises']
        separation = [.95] * len(cctypes)
        cctypes, distargs = cu.parse_distargs(cctypes)
        T, Zv, Zc = tu.gen_data_table(n_rows, view_weights, cluster_weights,
            cctypes, distargs, separation)
        T = T.T
        # Make some nan cells for evidence.
        T[5,2]=T[5,3]=T[5,0]=T[5,1]=np.nan
        T[8,4]=np.nan
        cls.engine = engine.Engine(T, cctypes, distargs, num_states=6,
            initialize=1)
        cls.engine.transition(N=2)

    def __ci_test_logpdf(self):
        for rowid in [-1, 5]:
            query1, evidence1 = [(0,1)], [(2,1), (3,.5)]
            query2, evidence2 = [(2,0), (5,3)], [(0,4), (1,5)]
            for Q, E in [(query1, evidence1), (query2, evidence2)]:
                # logpdfs should be a list of floats.
                logpdfs = self.engine.logpdf(rowid, Q, evidence=E)
                self.assertEqual(len(logpdfs), self.engine.num_states)
                for state_logpdfs in logpdfs:
                    # Each element in logpdfs should be a single float.
                    self.assertTrue(isinstance(state_logpdfs, float))

    def __ci_test_simulate(self):
        for rowid in [-1, 5]:
            for N in [1, 8]:
                query1, evidence1 = [0], [(2,0), (3,6)]
                query2, evidence2 = [1,2,5], [(0,3), (3,.8)]
                for Q, E in [(query1, evidence1), (query2, evidence2)]:
                    # samples should be a list of samples, one for each state.
                    samples = self.engine.simulate(rowid, Q, evidence=E, N=N)
                    self.assertEqual(len(samples), self.engine.num_states)
                    for states_samples in samples:
                        # Each element of samples should be a list of N samples.
                        self.assertEqual(len(states_samples), N)
                        for s in states_samples:
                            # Each raw sample should be len(Q) dimensional.
                            self.assertEqual(len(s), len(Q))

    def __ci_test_logpdf_bulk(self):
        rowid1, query1, evidence1 = 5, [(0,0), (5,3)], [(2,1), (3,.5)]
        rowid2, query2, evidence2 = -1, [(1,0), (4,.8)], [(5,.5)]
        # Bulk.
        rowids = [rowid1, rowid2]
        queries = [query1, query2]
        evidences = [evidence1, evidence2]
        # Invoke
        logpdfs = self.engine.logpdf_bulk(
            rowids, queries, evidences=evidences)
        self.assertEqual(len(logpdfs), self.engine.num_states)
        for state_logpdfs in logpdfs:
            # state_logpdfs should be a list of floats, one float per query.
            self.assertEqual(len(state_logpdfs), len(rowids))
            for l in state_logpdfs:
                    self.assertTrue(isinstance(l, float))

    def __ci_test_simulate_bulk(self):
        rowid1, query1, evidence1, N1, = -1, [0,2,4,5], [(3,1)], 7
        rowid2, query2, evidence2, N2 = 5, [1,3], [(2,.8)], 3
        rowid3, query3, evidence3, N3 = 8, [0], [(4,.8)], 3
        # Bulk.
        rowids = [rowid1, rowid2, rowid3]
        queries = [query1, query2, query3]
        evidences = [evidence1, evidence2, evidence3]
        Ns = [N1, N2, N3]
        # Invoke
        samples = self.engine.simulate_bulk(
            rowids, queries, evidences=evidences, Ns=Ns)
        self.assertEqual(len(samples), self.engine.num_states)
        for states_samples in samples:
            self.assertEqual(len(states_samples), len(rowids))
            for i, sample in enumerate(states_samples):
                self.assertEqual(len(sample), Ns[i])
                for s in sample:
                    self.assertEqual(len(s), len(queries[i]))

if __name__ == '__main__':
    unittest.main()
