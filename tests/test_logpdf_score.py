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

from cgpm.crosscat.engine import Engine
from cgpm.utils.general import gen_rng

def test_logpdf_score_crash():
    rng = gen_rng(8)
    # T = rng.choice([0,1], p=[.3,.7], size=250).reshape(-1,1)
    T = rng.normal(size=30).reshape(-1,1)
    engine = Engine(T, cctypes=['normal'], rng=rng, num_states=4)
    logpdf_likelihood_initial = np.array(engine.logpdf_likelihood())
    logpdf_score_initial = np.array(engine.logpdf_score())
    assert np.all(logpdf_score_initial < logpdf_likelihood_initial)
    # assert np.all(logpdf_likelihood_initial < logpdf_score_initial)
    engine.transition(N=100)
    engine.transition(kernels=['column_hypers','view_alphas'], N=10)
    logpdf_likelihood_final = np.asarray(engine.logpdf_likelihood())
    logpdf_score_final = np.asarray(engine.logpdf_score())
    assert np.all(logpdf_score_final < logpdf_likelihood_final)
    assert np.max(logpdf_score_initial) < np.max(logpdf_score_final)
