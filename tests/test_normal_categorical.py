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

"""This graphical test trains a gpmcc state on a bivariate population [X, Z].
X (called the data) is a cctype from DistributionGpm. Z is a categorical
variable that is a function of the latent cluster of each row
(called the indicator).

The three simulations are:
    - Joint Z,X.
    - Data conditioned on the indicator Z|X.
    - Indicator conditioned on the data X|Z.

Simulations are compared to synthetic data at various indicator subpopulations.
"""

import matplotlib.pyplot as plt
import numpy as np
import unittest

from scipy.stats import ks_2samp

import gpmcc.utils.general as gu
import gpmcc.utils.test as tu
from gpmcc.engine import Engine

class SimulateIndicatorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Entropy.
        cls.n_samples = 250
        # Generate synthetic data.
        T, Zv, Zc = tu.gen_data_table(cls.n_samples, [1], [[.3, .5, .2]],
            ['normal'], [None], [.95], rng=gu.gen_rng(0))
        cls.data = np.zeros((cls.n_samples, 2))
        cls.data[:,0] = T[0]
        cls.indicators = [0, 1, 2, 3, 4, 5]
        counts = {0:0, 1:0, 2:0}
        for i in xrange(cls.n_samples):
            k = Zc[0][i]
            cls.data[i,1] = 2*cls.indicators[k] + counts[k] % 2
            counts[k] += 1
        # Create an engine.
        state = Engine(cls.data, ['normal', 'categorical'], [None, {'k':6}],
            num_states=1, initialize=True)
        state.transition(N=200)
        cls.model = state.get_state(0)

    def test_joint__ci_(self):
        # Simulate from the joint distribution of (x,i).
        joint_samples = self.model.simulate(-1, [0,1], N=self.n_samples)
        _, ax = plt.subplots()
        ax.set_title('Joint Simulation')
        for t in self.indicators:
            # Plot original data.
            data_subpop = self.data[self.data[:,1] == t]
            ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
            # Plot simulated data.
            joint_samples_subpop = joint_samples[joint_samples[:,1] == t]
            ax.scatter(joint_samples_subpop[:,1] + .25,
                joint_samples_subpop[:,0], color=gu.colors[t])
            # KS test.
            pvalue = ks_2samp(data_subpop[:,0], joint_samples_subpop[:,0])[1]
            self.assertGreater(pvalue, 0.05)
        ax.set_xlabel('Indicator')
        ax.set_ylabel('x')
        ax.grid()

    def test_conditional_indicator__ci_(self):
        # Simulate from the conditional X|Z
        _, ax = plt.subplots()
        ax.set_title('Conditional Simulation Of Data X Given Indicator Z')
        for t in self.indicators:
            # Plot original data.
            data_subpop = self.data[self.data[:,1] == t]
            ax.scatter(data_subpop[:,1], data_subpop[:,0], color=gu.colors[t])
            # Plot simulated data.
            conditional_samples_subpop = self.model.simulate(-1, [0],
                evidence=[(1,t)], N=len(data_subpop))
            ax.scatter(np.repeat(t, len(data_subpop)) + .25,
                conditional_samples_subpop[:,0], color=gu.colors[t])
            # KS test.
            pvalue = ks_2samp(data_subpop[:,0], conditional_samples_subpop[:,0])[1]
            self.assertGreater(pvalue, 0.1)
        ax.set_xlabel('Indicator')
        ax.set_ylabel('x')
        ax.grid()

    def test_conditional_real__ci_(self):
        # Simulate from the conditional Z|X
        fig, axes = plt.subplots(2,3)
        fig.suptitle('Conditional Simulation Of Indicator Z Given Data X')
        # Compute representative data sample for each dindicator.
        means = [np.mean(self.data[self.data[:,1]==i], axis=0)[0] for
            i in self.indicators]
        for mean, indicator, ax in zip(means, self.indicators, axes.ravel('F')):
            conditional_samples_subpop = self.model.simulate(-1, [1],
                evidence=[(0,mean)], N=self.n_samples)
            ax.hist(conditional_samples_subpop, color='g', alpha=.4)
            ax.set_title('True Indicator %d' % indicator)
            ax.set_xlabel('Simulated Indicator')
            ax.set_xticks(self.indicators)
            ax.set_ylabel('Frequency')
            ax.set_ylim([0, ax.get_ylim()[1]+10])
            ax.grid()

if __name__ == '__main__':
    unittest.main()
