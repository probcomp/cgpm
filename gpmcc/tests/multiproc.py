# -*- coding: utf-8 -*-

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from gpmcc.utils import test as tu
from gpmcc import engine
from gpmcc.state import State
import numpy as np

# This script generates a column of every data type and plots inference
# in real time.

# Set up the data generation.
n_rows = 200
view_weights = np.asarray([0.7, .3])
cluster_weights = [np.array([.33, .33, .34]), np.array([.2, .8])]
cctypes = ['beta_uc', 'normal','normal_uc','poisson','categorical',
    'vonmises', 'bernoulli', 'lognormal']

separation = [.7] * 9
distargs = [None, None, None, None, {'k':5}, None, None, None, None]

T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation, return_dims=True)

runner = engine.Engine()
runner.initialize(T, cctypes, distargs, num_states=6)
runner.transition(N=2)

# state = State(T, cctypes, distargs)
# state.transition(N=10, do_plot=True)
