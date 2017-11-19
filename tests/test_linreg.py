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

import importlib
import pytest

import matplotlib.pyplot as plt
import numpy as np

from cgpm.regressions.linreg import LinearRegression
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


CCTYPES, DISTARGS = cu.parse_distargs([
    'normal',
    'categorical(k=4)',
    'lognormal',
    'poisson',
    'bernoulli',
    'exponential',
    'geometric',
    'vonmises'])

D, Zv, Zc = tu.gen_data_table(
    50, [1], [[.33, .33, .34]], CCTYPES, DISTARGS, [.8]*len(CCTYPES),
    rng=gu.gen_rng(0))

CCTYPES = CCTYPES[1:]
CCARGS = DISTARGS[1:]
OUTPUTS = [0]
INPUTS = range(1, len(CCTYPES)+1)
CCARGS[CCTYPES.index('bernoulli')] = {'k':2}
D = D.T


def test_incorporate():
    linreg = LinearRegression(
        OUTPUTS, INPUTS,
        distargs={'inputs':{'stattypes': CCTYPES, 'statargs': CCARGS}},
        rng=gu.gen_rng(0))
    # Incorporate first 20 rows.
    for rowid, row in enumerate(D[:20]):
        observation = {0: row[0]}
        inputs = {i:row[i] for i in linreg.inputs}
        linreg.incorporate(rowid, observation, inputs)
    # Unincorporating row 20 should raise.
    with pytest.raises(ValueError):
        linreg.unincorporate(20)
    # Unincorporate all rows.
    for rowid in xrange(20):
        linreg.unincorporate(rowid)
    # Unincorporating row 0 should raise.
    with pytest.raises(ValueError):
        linreg.unincorporate(0)
    # Incorporating with wrong covariate dimensions should raise.
    with pytest.raises(ValueError):
        observation = {0: D[0,0]}
        inputs = {i:v for (i, v) in enumerate(D[0])}
        linreg.incorporate(0, observation, inputs)
    # Incorporating with missing observation should raise.
    with pytest.raises(ValueError):
        observation = {0: None}
        inputs = {i:v for (i, v) in enumerate(D[0])}
        linreg.incorporate(0, observation, inputs)
    # Incorporating with wrong observation should raise.
    with pytest.raises(ValueError):
        observation = {1: 2}
        inputs = {i:v for (i, v) in enumerate(D[0])}
        linreg.incorporate(0, observation, inputs)
    # Incorporate some more rows.
    for rowid, row in enumerate(D[:10]):
        observation = {0: row[0]}
        inputs = {i:row[i] for i in linreg.inputs}
        linreg.incorporate(rowid, observation, inputs)


def test_logpdf_score():
    linreg = LinearRegression(
        OUTPUTS, INPUTS,
        distargs={'inputs':{'stattypes': CCTYPES, 'statargs': CCARGS}},
        rng=gu.gen_rng(0))
    for rowid, row in enumerate(D[:10]):
        observation = {0: row[0]}
        inputs = {i:row[i] for i in linreg.inputs}
        linreg.incorporate(rowid, observation, inputs)
    linreg.transition_hypers(N=10)
    assert linreg.logpdf_score() < 0


def test_logpdf_predictive():
    linreg = LinearRegression(
        OUTPUTS, INPUTS,
        distargs={'inputs':{'stattypes': CCTYPES, 'statargs': CCARGS}},
        rng=gu.gen_rng(0))
    Dx0 = D[D[:,1]==0]
    Dx1 = D[D[:,1]==1]
    Dx2 = D[D[:,1]==2]
    Dx3 = D[D[:,1]==3]
    for i, row in enumerate(Dx0[1:]):
        linreg.incorporate(i, {0: row[0]}, {i: row[i] for i in linreg.inputs})
    linreg.transition_hypers(N=10)
    # Ensure can compute predictive for seen class 0.
    linreg.logpdf(None, {0: Dx0[0,0]},
        inputs={i: Dx0[0,i] for i in linreg.inputs})
    # Ensure can compute predictive for unseen class 1.
    linreg.logpdf(None, {0: Dx1[0,0]},
        inputs={i: Dx1[0,i] for i in linreg.inputs})
    # Ensure can compute predictive for unseen class 2.
    linreg.logpdf(None, {0: Dx2[0,0]},
        inputs={i: Dx2[0,i] for i in linreg.inputs})
    # Ensure can compute predictive for unseen class 3.
    linreg.logpdf(None, {0: Dx3[0,0]},
        inputs={i: Dx3[0,i] for i in linreg.inputs})
    # Ensure can compute predictive for nan.
    with pytest.raises(ValueError):
        linreg.logpdf(None, {0: np.nan},
            inputs={i: Dx0[0,i] for i in linreg.inputs})
    # Ensure can compute predictive for missing targets.
    with pytest.raises(ValueError):
        linreg.logpdf(None, {7: 10},
            inputs={i: Dx0[0,i] for i in linreg.inputs})

def test_simulate():
    linreg = LinearRegression(
        OUTPUTS, INPUTS,
        distargs={'inputs':{'stattypes': CCTYPES, 'statargs': CCARGS}},
        rng=gu.gen_rng(0))
    for rowid, row in enumerate(D[:25]):
        linreg.incorporate(rowid, {0:row[0]}, {i:row[i] for i in linreg.inputs})
    linreg.transition_hypers(N=10)

    # Use a deserialized version for simulating.
    metadata = linreg.to_metadata()
    builder = getattr(
        importlib.import_module(metadata['factory'][0]),
        metadata['factory'][1])
    linreg = builder.from_metadata(metadata, rng=gu.gen_rng(1))

    _, ax = plt.subplots()
    xpred, xtrue = [], []
    for row in D[25:]:
        xtrue.append(row[0])
        inputs = {i: row[i] for i in linreg.inputs}
        samples = [linreg.simulate(None, [0], None, inputs)[0]
            for _i in xrange(100)]
        xpred.append(samples)
    xpred = np.asarray(xpred)
    xmeans = np.mean(xpred, axis=1)
    xlow = np.percentile(xpred, 25, axis=1)
    xhigh = np.percentile(xpred, 75, axis=1)
    ax.plot(range(len(xtrue)), xmeans, color='g')
    ax.fill_between(range(len(xtrue)), xlow, xhigh, color='g', alpha='.3')
    ax.scatter(range(len(xtrue)), xtrue, color='r')
    # plt.close('all')


def test_missing_inputs():
    outputs = [0]
    inputs = [2, 4, 6]
    distargs = {
        'inputs': {
            'stattypes': ['normal', 'categorical', 'categorical'],
            'statargs': [None, {'k': 4}, {'k': 1}]
        },
    }
    linreg = LinearRegression(
        outputs=outputs,
        inputs=inputs,
        distargs=distargs,
        rng=gu.gen_rng(1))

    # Incorporate invalid cateogry 4:100. The first term is the bias, the second
    # terms is {2:1}, the next four terms are the dummy code of {4:100}, and the
    # last term is the code for {6:0}
    rowid = 0
    with pytest.raises(ValueError):
        linreg.incorporate(rowid, {0:1}, {2:1, 4:100, 6:0})

    # Incorporate invalid cateogry 6:1. The first term is the bias, the second
    # terms is {2:5}, the next four terms are the dummy code of {4:3}, and the
    # last term is the code for {6:0}. Since linreg has a wildcard category
    # for composition with the CRP mixture, should be handled without error.
    rowid = 1
    linreg.incorporate(rowid, {0:2}, {2:5, 4:3, 6:1})
    assert linreg.data.Y[rowid] == [1, 5, 0, 0, 0, 1, 0]

    # Incorporate missing cateogry for input 6. The first term is the bias, the
    # second terms is {2:5}, the next four terms are the dummy code of {4:0}
    # and the last term is the code for {6:missing}.
    rowid = 2
    with pytest.raises(ValueError):
        linreg.incorporate(rowid, {0:5}, {2:6, 4:0})

    # Missing input 2 should be imputed to av(1,5,7) == 4.
    rowid = 3
    with pytest.raises(ValueError):
        linreg.incorporate(rowid, {0:4}, {4:1, 6:0})

    linreg.transition_hypers(N=10)
