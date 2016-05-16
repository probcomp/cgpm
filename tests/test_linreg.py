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

import matplotlib.pyplot as plt
import numpy as np

from gpmcc.dists.linreg import LinearRegression
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


class LinearRegressionDirectTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cctypes, cls.distargs = cu.parse_distargs(['normal',
            'categorical(k=4)','lognormal','poisson','bernoulli',
            'exponential','geometric','vonmises'])
        D, Zv, Zc = tu.gen_data_table(50, [1], [[.33, .33, .34]], cls.cctypes,
            cls.distargs, [.8]*len(cls.cctypes), rng=gu.gen_rng(0))
        cls.cctypes = cls.cctypes[1:]
        cls.ccargs = cls.distargs[1:]
        cls.outputs = [0]
        cls.inputs = range(1, len(cls.cctypes)+1)
        cls.ccargs[cls.cctypes.index('bernoulli')] = {'k':2}
        cls.D = D.T

    def test_incorporate(self):
        linreg = LinearRegression(
            self.outputs, self.inputs,
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        # Incorporate first 20 rows.
        for rowid, row in enumerate(self.D[:20]):
            query = {0: row[0]}
            evidence = {i:row[i] for i in linreg.inputs}
            linreg.incorporate(rowid, query, evidence)
        # Unincorporating row 20 should raise.
        with self.assertRaises(KeyError):
            linreg.unincorporate(20)
        # Unincorporate all rows.
        for rowid in xrange(20):
            linreg.unincorporate(rowid)
        # Unincorporating row 0 should raise.
        with self.assertRaises(KeyError):
            linreg.unincorporate(0)
        # Incorporating with wrong covariate dimensions should raise.
        with self.assertRaises(TypeError):
            query = {0: self.D[0,0]}
            evidence = {i:v for (i, v) in enumerate(self.D[0])}
            linreg.incorporate(0, query, evidence)
        # Incorporate some more rows.
        for rowid, row in enumerate(self.D[:10]):
            query = {0: row[0]}
            evidence = {i:row[i] for i in linreg.inputs}
            linreg.incorporate(rowid, query, evidence)

    def test_logpdf_score(self):
        linreg = LinearRegression(
            self.outputs, self.inputs,
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        for rowid, row in enumerate(self.D[:10]):
            query = {0: row[0]}
            evidence = {i:row[i] for i in linreg.inputs}
            linreg.incorporate(rowid, query, evidence)
        self.assertLess(linreg.logpdf_score(), 0)

    def test_logpdf_predictive(self):
        linreg = LinearRegression(
            self.outputs, self.inputs,
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        Dx0 = self.D[self.D[:,1]==0]
        Dx1 = self.D[self.D[:,1]==1]
        Dx2 = self.D[self.D[:,1]==2]
        Dx3 = self.D[self.D[:,1]==3]
        for i, row in enumerate(Dx0[1:]):
            linreg.incorporate(
                i, {0: row[0]}, {i:row[i] for i in linreg.inputs})
        # Ensure can compute predictive for seen class 0.
        linreg.logpdf(
            -1, {0: Dx0[0,0]}, {i:Dx0[0,i] for i in linreg.inputs})
        # Ensure can compute predictive for unseen class 1.
        linreg.logpdf(
            -1, {0: Dx1[0,0]}, {i:Dx1[0,i] for i in linreg.inputs})
        # Ensure can compute predictive for unseen class 2.
        linreg.logpdf(
            -1, {0: Dx2[0,0]}, {i:Dx2[0,i] for i in linreg.inputs})
        # Ensure can compute predictive for unseen class 3.
        linreg.logpdf(-1, {0: Dx3[0,0]}, {i:Dx3[0,i] for i in linreg.inputs})

    def test_simulate(self):
        linreg = LinearRegression(
            self.outputs, self.inputs,
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        for rowid, row in enumerate(self.D[:25]):
            query = {0: row[0]}
            evidence = {i:row[i] for i in linreg.inputs}
            linreg.incorporate(rowid, query, evidence)
        _, ax = plt.subplots()
        xpred, xtrue = [], []
        for row in self.D[25:]:
            xtrue.append(row[0])
            xpred.append(
                [linreg.simulate(-1, [0], {i:row[i] for i in linreg.inputs})
                for _ in xrange(100)])
        xpred = np.asarray(xpred)
        xmeans = np.mean(xpred, axis=1)
        xlow = np.percentile(xpred, 25, axis=1)
        xhigh = np.percentile(xpred, 75, axis=1)
        ax.plot(range(len(xtrue)), xmeans, color='g')
        ax.fill_between(range(len(xtrue)), xlow, xhigh, color='g', alpha='.3')
        ax.scatter(range(len(xtrue)), xtrue, color='r')
        # plt.close('all')

if __name__ == '__main__':
    unittest.main()
