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

"""
This test suite ensures that results returned from View.logpdf_multirow,
State.logpdf_multirow and Engine.logpdf_multirow, are analytically correct
(for some cases) and follow the rules of probability. 
"""
# RUN TESTS IN cgpm/ DIRECTORY
 
import pytest
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu
from cgpm.utils.general import logsumexp

from cgpm.mixtures.view import View

OUT = 'tests/resources/out/'

@pytest.fixture(params=[(seed, D) for seed in range(3) for D in [1]])
def priorCGPM(request):
    seed = request.param[0]
    D = request.param[1]
    rng = gu.gen_rng(seed)
    data = rng.choice([0, 1], size=(100, D))
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    model = View(
        X,
        cctypes=['bernoulli']*D,
        outputs=[1000] + outputs,
        rng=rng)
    model.seed = seed
    return model

def analyze(model):
    model.transition(N=10)
    return model

def test_logpdf_multirow_plot(priorCGPM):
    view = priorCGPM

    D = len(view.outputs)-1
    query = {0: 1 for i in range(D)}
    d = {} 
    logp = [view.logpdf(-1, query=query)]
    for i in range(5):
        d[i] = {i: 1 for i in range(D)}
        logp.append(view.logpdf_multirow(-1, query=query, evidence=d))
    Nx = range(len(logp))
    plt.plot(Nx, logp)
    plt.xlabel("$n$ (number of conditioned rows)")
    plt.ylabel("$\log(p(x_t=1|x_1=1,\ldots,x_n=1))$")
    plt.savefig(OUT + '''
    test_logpdf_multirow_seed%d_%dD_view.png'''
                % (priorCGPM.seed, D))
    plt.clf()

