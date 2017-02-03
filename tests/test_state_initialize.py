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

from cgpm.crosscat.state import State
from cgpm.utils import general as gu

def test_Zv_without_Zrv():
    rng = gu.gen_rng(2)
    D = rng.normal(size=(10,4))

    state = State(
        D,
        outputs=[3,2,1,0,],
        cctypes=['normal']*D.shape[1],
        Zv={3:0, 2:1, 1:2, 0:4},
        rng=rng,
    )
