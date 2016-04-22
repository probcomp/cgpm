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
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gpmcc.dists.linreg import LinearRegression

from gpmcc.state import State
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
        cls.ccargs[cls.cctypes.index('bernoulli')] = {'k':2}
        cls.D = D.T

    def test_incorporate(self):
        linreg = LinearRegression(
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        # Incorporate first 20 rows.
        for row in self.D[:20]:
            linreg.incorporate(row[0], y=row[1:])
        # Unincorporating row 20 should raise.
        with self.assertRaises(ValueError):
            linreg.unincorporate(self.D[20,0], y=self.D[20,1:])
        # Unincorporate all rows.
        for row in self.D[:20]:
            linreg.unincorporate(row[0], y=row[1:])
        # Unincorporating row 0 should raise.
        with self.assertRaises(ValueError):
            linreg.unincorporate(self.D[0,0], y=self.D[0,1:])
        # Incorporating with wrong covariate dimensions should raise.
        with self.assertRaises(TypeError):
            linreg.incorporate(self.D[0,0], y=self.D[0,:])
        # Incorporate some more rows.
        for row in self.D[:10]:
            linreg.incorporate(row[0], y=row[1:])

    def test_logpdf_marginal(self):
        linreg = LinearRegression(
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        for row in self.D[:25]:
            linreg.incorporate(row[0], y=row[1:])
        self.assertLess(linreg.logpdf_marginal(), 0)

    def test_logpdf_predictive(self):
        linreg = LinearRegression(
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        Dx0 = self.D[self.D[:,1]==0]
        Dx1 = self.D[self.D[:,1]==1]
        Dx2 = self.D[self.D[:,1]==2]
        Dx3 = self.D[self.D[:,1]==3]
        for row in Dx0[:-1]:
            linreg.incorporate(row[0], y=row[1:])
        # Ensure can compute predictive for seen class 0.
        linreg.logpdf(Dx0[-1,0], y=Dx0[-1,1:])
        # Ensure can compute predictive for unseen class 1.
        linreg.logpdf(Dx1[0,0], y=Dx1[0,1:])
        # Ensure can compute predictive for unseen class 2.
        linreg.logpdf(Dx2[0,0], y=Dx2[0,1:])
        # Ensure can compute predictive for unseen class 3.
        linreg.logpdf(Dx3[0,0], y=Dx3[0,1:])

    def test_simulate(self):
        linreg = LinearRegression(
            distargs={'cctypes':self.cctypes, 'ccargs':self.ccargs},
            rng=gu.gen_rng(0))
        for row in self.D[:25]:
            linreg.incorporate(row[0], y=row[1:])
        _, ax = plt.subplots()
        xpred, xtrue = [], []
        for row in self.D[25:]:
            xtrue.append(row[0])
            xpred.append([linreg.simulate(y=row[1:]) for _ in xrange(100)])
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
