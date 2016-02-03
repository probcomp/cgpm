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

import gpmcc.utils.general as gu
import gpmcc.utils.test as tu
from gpmcc.engine import Engine

class SimulateIndicatorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Entropy.
        np.random.seed(0)
        cls.n_samples = 250
        n_transitions = 50
        # Generate synthetic data.
        view_weights = [1.]
        cluster_weights = [[.3, .5, .2]]
        cctypes = ['normal']
        separation = [.95]
        T, Zv, Zc = tu.gen_data_table(cls.n_samples, view_weights,
            cluster_weights, cctypes, [None], separation)
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
        state.transition(N=n_transitions)
        cls.model = state.get_state(0)

    def test_joint(self):
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
        ax.set_xlabel('Indicator')
        ax.set_ylabel('x')
        ax.grid()

    def test_conditional_indicator(self):
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
        ax.set_xlabel('Indicator')
        ax.set_ylabel('x')
        ax.grid()

    def test_conditional_real(self):
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
        # plt.close('all')

if __name__ == '__main__':
    unittest.main()
