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

import pytest

import numpy as np

from gpmcc.state import State
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu

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

import pytest

import numpy as np

from gpmcc.state import State
from gpmcc.utils import config as cu
from gpmcc.utils import general as gu
from gpmcc.utils import test as tu


CCTYPES, DISTARGS = cu.parse_distargs([
    'normal',        # 0
    'poisson',       # 1
    'bernoulli',     # 2
    'lognormal',     # 3
    'exponential',   # 4
    'geometric',])   # 5


T, Zv, Zc = tu.gen_data_table(
    20, [1], [[.33, .33, .34]], CCTYPES, DISTARGS,
    [.95]*len(CCTYPES), rng=gu.gen_rng(0))
T = T.T


def test_incorporate():
    state = State(
        T[:5,:], cctypes=CCTYPES, distargs=DISTARGS, rng=gu.gen_rng(0))
    state.transition(N=5)

    # Incorporate row into cluster 0 for all views.
    # previous = np.asarray([v.Nk[0] for v in state.views])
    # data = {i: T[6,i] for i in xrange(6)}
    # state.incorporate_row(-1, data, k=[0]*len(state.views))
    # assert [v.Nk[0] for v in state.views] == list(previous+1)

    # Incorporate row into a singleton for all views.
    # previous = [len(v.Nk) for v in state.views]
    # data = {i: T[11,i] for i in xrange(6)}
    # state.incorporate_row(11, data, k=previous)
    # assert [len(v.Nk) for v in state.views] == [p+1 for p in previous]

    # Unincorporate row from the singleton view just created.
    # previous = [len(v.Nk) for v in state.views]
    # state.unincorporate_row(11)
    # assert [len(v.Nk) for v in state.views] == [p-1 for p in previous]

    # # Undo the last step.
    # previous = np.asarray([len(v.Nk) for v in state.views])
    # state.incorporate_rows([T[11,:]], k=[previous])
    # assert [len(v.Nk) for v in state.views] == list(previous+1)

    # # Incorporate row without specifying a view.
    # state.incorporate_rows([T[12,:]], k=None)

    # # Incorporate row specifying different clusters.
    # k = [None] * len(state.views)
    # k[::2] = [1] * len(k[::2])
    # previous = np.asarray([v.Nk[1] for v in state.views])
    # state.incorporate_rows([T[13,:]], k=[k])
    # for i in xrange(len(state.views)):
    #     if i%2 == 0:
    #         assert state.views[i].Nk[1] == previous[i]+1

    # # Incorporate two rows with different clusterings.
    # previous = np.asarray([v.Nk[0] for v in state.views])
    # k = [0 for _ in xrange(len(state.views))]
    # state.incorporate_rows(T[[14,15],:], k=[k,k])
    # assert state.views[0].Nk[0] == previous[0]+2

    # # Incoporate remaining rows in the default way.
    # state.incorporate_rows(T[16:,:])
    # assert state.n_rows() == len(T)

    # # Unincorporate all rows except the last one using ascending.
    # state.unincorporate_rows(xrange(1, len(T)))
    # assert state.n_rows() == 1

    # # Reincorporate all rows.
    # state.incorporate_rows(T[1:,:])
    # assert state.n_rows() == len(T)

    # # Unincorporate all rows except the last one using descending.
    # state.unincorporate_rows(xrange(len(T)-1, 0, -1))
    # assert state.n_rows() == 1

    # # Unincorporating last dim should raise.
    # with pytest.raises(ValueError):
    #     state.unincorporate_rows([0])

