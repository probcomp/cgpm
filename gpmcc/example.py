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

from gpmcc.utils import config as cu
from gpmcc.utils import test as tu
from gpmcc import state
import numpy

# This script generates acolumn of every data type and plots inference
# in real time

numpy.random.seed(10)

# set up the data generation
n_rows = 200
view_weights = numpy.ones(1)
cluster_weights = [ numpy.array([.33, .33, .34]) ]
cctypes = ['normal','poisson','bernoulli', 'lognormal', 'exponential','normal_uc',
'beta_uc', 'exponential_uc','geometric']
separation = [.95] * len(cctypes)
cctypes, distargs = cu.parse_distargs(cctypes)

T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation, return_dims=True)

S = state.State(T.T, cctypes, distargs, seed=0)
S.transition(N=30)
S.transition(N=1, do_plot=True)
