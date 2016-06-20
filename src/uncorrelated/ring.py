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

import numpy as np

from cgpm.uncorrelated.undirected import UnDirectedXyGpm


class Ring(UnDirectedXyGpm):
    """sqrt(X**2 + Y**2) + noise = 1 """

    def simulate_joint(self):
        angle = self.rng.uniform(0., 2.*np.pi)
        distance = self.rng.uniform(1.-self.noise, 1.)
        x = np.cos(angle)*distance
        y = np.sin(angle)*distance
        return [x, y]
