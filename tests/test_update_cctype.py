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

from gpmcc.state import State
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu

class UpdateCctypeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cctypes, cls.distargs = cu.parse_distargs(['normal','poisson',
            'categorical(k=2)','bernoulli','lognormal','exponential',
            'geometric','vonmises'])
        T, Zv, Zc = tu.gen_data_table(20, [1], [[.33, .33, .34]], cls.cctypes,
            cls.distargs, [.95]*len(cls.cctypes), rng=gu.gen_rng(0))
        cls.T = T.T

    def test_categorical_bernoulli(self):
        state = State(self.T, self.cctypes, distargs=self.distargs,
            rng=gu.gen_rng(0))
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('categorical'), 'bernoulli')
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('categorical'), 'categorical', distargs={'k':2})

    def test_poisson_categorical(self):
        state = State(self.T, self.cctypes, distargs=self.distargs,
            rng=gu.gen_rng(0))
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('categorical'), 'poisson')
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('categorical'), 'categorical', distargs={'k':2})

    def test_vonmises_normal(self):
        state = State(self.T, self.cctypes, distargs=self.distargs,
            rng=gu.gen_rng(0))
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('vonmises'), 'normal')
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('vonmises'), 'vonmises')
        with self.assertRaises(Exception):
            state.update_cctype(
                self.cctypes.index('normal'), 'vonmises')

    def test_geometric_exponential(self):
        state = State(self.T, self.cctypes, distargs=self.distargs,
            rng=gu.gen_rng(0))
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('geometric'), 'exponential')
        state.transition(N=1)
        with self.assertRaises(Exception):
            state.update_cctype(
                self.cctypes.index('exponential'), 'geometric')

    def test_categorical_forest(self):
        state = State(self.T, self.cctypes, distargs=self.distargs,
            rng=gu.gen_rng(0))
        state.transition(N=1)
        state.update_cctype(
            self.cctypes.index('categorical'), 'random_forest',
            distargs={'k':2})

if __name__ == '__main__':
    unittest.main()
