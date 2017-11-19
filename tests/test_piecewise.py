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

from cgpm.dummy.piecewise import PieceWise
from cgpm.utils.general import logsumexp


def test_piecewise_logpdf():
    pw = PieceWise([0,1], [2], sigma=1, flip=.8)
    # x,z
    pw.simulate(None, [0,1], None, {2:1})
    pw.logpdf(None, {0:1.5, 1:0}, None, {2:1})

    # x
    pw.simulate(None, [0], None, {2:1})
    pw.logpdf(None, {0:1.5}, None, {2:1})

    # z
    pw.simulate(None, [1], None, {2:1})
    assert np.allclose(
        logsumexp([
            pw.logpdf(None, {1:0}, None, {2:1}),
            pw.logpdf(None, {1:1}, None, {2:1})]),
        0)

    # z|x
    pw.simulate(None, [1], {0:1.5}, {2:1})
    assert np.allclose(
        logsumexp([
            pw.logpdf(None, {1:0}, {0:1.5}, {2:1}),
            pw.logpdf(None, {1:1}, {0:1.5}, {2:1})]),
        0)

    # x|z
    pw.simulate(None, [0], {1:0}, {2:1})
    pw.logpdf(None, {0:1.5}, {1:0}, {2:1})
