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

from builtins import range
from cgpm.crosscat.engine import Engine
from cgpm.utils import general as gu

def test_engine_simulate_no_repeat():
    """Generate 3 samples from 2 states 10 times, and ensure uniqueness."""
    rng = gu.gen_rng(1)
    engine = Engine(X=[[1]], cctypes=['normal'], num_states=2, rng=rng)
    samples_list = [
        [sample[0] for sample in engine.simulate(None, [0], N=3)[0]]
        for _i in range(10)
    ]
    samples_set = set([frozenset(s) for s in samples_list])
    assert len(samples_set) == len(samples_list)
