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

from math import log

import numpy as np

from cgpm.regressions.ols import OrdinaryLeastSquares
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


cctypes, distargs = cu.parse_distargs([
    'normal',
    'categorical(k=3)',
    'poisson',
    'bernoulli',
    'lognormal',
    'exponential',
    'geometric',
    'vonmises',
    'normal'])

T, Zv, Zc = tu.gen_data_table(
    100, [1], [[.33, .33, .34]], cctypes, distargs,
    [.2]*len(cctypes), rng=gu.gen_rng(0))

D = T.T
OLS_DISTARGS = {
    'inputs': {
        'stattypes': cctypes[1:],
        'statargs':
            [{'k': 3}] + [None] + [{'k': 2}] + [None, None, None, None, None]
    }
}
OLS_OUTPUTS = [0]
OLS_INPUTS = range(1, len(cctypes))


def test_integration():
    ols = OrdinaryLeastSquares(
        outputs=OLS_OUTPUTS, inputs=OLS_INPUTS,
        distargs=OLS_DISTARGS, rng=gu.gen_rng(0))

    # Incorporate first 20 rows.
    for rowid, row in enumerate(D[:20]):
        query = {0: row[0]}
        evidence = {i: row[i] for i in ols.inputs}
        ols.incorporate(rowid, query, evidence)
    # Unincorporating row 20 should raise.
    with pytest.raises(ValueError):
        ols.unincorporate(20)
    # Unincorporate all rows.
    for rowid in xrange(20):
        ols.unincorporate(rowid)
    # Unincorporating row 0 should raise.
    with pytest.raises(ValueError):
        ols.unincorporate(0)
    # Incorporating with wrong covariate dimensions should raise.
    with pytest.raises(ValueError):
        query = {0: D[0,0]}
        evidence = {i: v for (i, v) in enumerate(D[0])}
        ols.incorporate(0, query, evidence)
    # Incorporating with None output value should raise.
    with pytest.raises(ValueError):
        query = {0: None}
        evidence = {i: D[0,i] for i in ols.inputs}
        ols.incorporate(0, query, evidence)
    # Incorporating with nan evidence value should raise.
    with pytest.raises(ValueError):
        query = {0: 100}
        evidence = {i: D[0,i] for i in ols.inputs}
        evidence[evidence.keys()[0]] = np.nan
        ols.incorporate(0, query, evidence)
    # Incorporate some more rows.
    for rowid, row in enumerate(D[:10]):
        query = {0: row[0]}
        evidence = {i: row[i] for i in ols.inputs}
        ols.incorporate(rowid, query, evidence)

    # Run a transition.
    ols.transition()
    assert ols.noise > 0

    # Invalid categorical evidence 5 for categorical(k=3).
    query = {OLS_OUTPUTS[0]: 2}
    evidence = dict(zip(OLS_INPUTS, [5, 5, 0, 1.4, 7, 4, 2, -2]))
    with pytest.raises(ValueError):
        ols.logpdf(-1, query, evidence)
    with pytest.raises(ValueError):
        ols.simulate(-1, OLS_OUTPUTS, evidence)

    # Invalid categorical evidence 2 for bernoulli.
    query = {OLS_OUTPUTS[0]: 2}
    evidence = dict(zip(OLS_INPUTS, [5, 5, 2, 1.4, 7, 4, 2, -2]))
    with pytest.raises(ValueError):
        ols.logpdf(-1, query, evidence)
    with pytest.raises(ValueError):
        ols.simulate(-1, OLS_OUTPUTS, evidence)

    # Do a logpdf computation.
    query = {OLS_OUTPUTS[0]: 2}
    evidence = dict(zip(OLS_INPUTS, [2, 5, 0, 1.4, 7, 4, 2, -2]))
    logp_old = ols.logpdf(-1, query, evidence)
    assert logp_old < 0
    ols.simulate(-1, OLS_OUTPUTS, evidence)

    # Now serialize and deserialize, and check if logp_old is the same.
    metadata = ols.to_metadata()
    builder = getattr(
        importlib.import_module(metadata['factory'][0]),
        metadata['factory'][1])
    ols2 = builder.from_metadata(metadata, rng=gu.gen_rng(1))

    assert ols2.noise == ols.noise
    logp_new = ols2.logpdf(-1, query, evidence)
    assert np.allclose(logp_new, logp_old)
    ols2.simulate(-1, OLS_OUTPUTS, evidence)
