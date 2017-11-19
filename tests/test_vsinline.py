# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
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

import hacks
import pytest
if not pytest.config.getoption('--integration'):
    hacks.skip('specify --integration to run integration tests')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cgpm.venturescript.vsinline import InlineVsCGpm
from cgpm.utils import general as gu


def test_input_matches_args():
    InlineVsCGpm([0], [], expression='() ~> {normal(0, 1)}')
    InlineVsCGpm([0], [], expression='( ) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [], expression='()   ~> {normal(0, 1)}')
    InlineVsCGpm([0], [], expression=' ( ) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1], expression='(a) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1], expression='(a ) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1], expression=' ( a ) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1], expression='( ab) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1,2], expression='(a, b) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1,2], expression='( a, b ) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1,2], expression='(a, b ) ~> {normal(0, 1)}')
    InlineVsCGpm([0], [1,2,3], expression='(a, b, bc) ~> {2}')

    with pytest.raises(Exception):
        InlineVsCGpm([0], [], expression='(a) ~> {normal(0,1)}')
    with pytest.raises(Exception):
        InlineVsCGpm([0], [1], expression='(a, b) ~> {normal(0,1)}')
    with pytest.raises(Exception):
        InlineVsCGpm([0], [4], expression='(a, b , c) ~> {normal(0,1)}')
    with pytest.raises(Exception):
        InlineVsCGpm([0], [1,2], expression='(a) ~> {normal(0,1)}')


def test_simulate_uniform():
    vs = InlineVsCGpm([0], [],
        expression='() ~> {uniform(low: -4.71, high: 4.71)}',
        rng=gu.gen_rng(10))

    lp = vs.logpdf(0, {0:0})
    for x in np.linspace(-4.70, 4.70, 100):
        assert np.allclose(vs.logpdf(0, {0:x}), lp)
    assert np.isinf(vs.logpdf(0, {0:12}))

    samples = vs.simulate(0, [0], None, None, N=200)
    extracted = [s[0] for s in samples]
    fig, ax = plt.subplots()
    ax.hist(extracted)


def test_simulate_noisy_cos():
    vs_x = InlineVsCGpm([0], [],
        expression='() ~> {uniform(low: -4.71, high: 4.71)}',
        rng=gu.gen_rng(10))

    vs_y = InlineVsCGpm([1], [0],
        expression="""
        (x) ~>
            {if (cos(x) > 0)
                {uniform(low: cos(x) - .5, high: cos(x))}
            else
                {uniform(low: cos(x), high: cos(x) + .5)}}""",
        rng=gu.gen_rng(12))

    samples_x = vs_x.simulate(0, [0], None, None, N=200)
    samples_y = [vs_y.simulate(0, [1], None, sx) for sx in samples_x]

    # Plot the joint query.
    fig, ax = plt.subplots()

    xs = [s[0] for s in samples_x]
    ys = [s[1] for s in samples_y]

    # Scatter the dots.
    ax.scatter(xs, ys, color='blue', alpha=.4)
    ax.set_xlim([-1.5*np.pi, 1.5*np.pi])
    ax.set_ylim([-1.75, 1.75])
    for x in xs:
        ax.vlines(x, -1.75, -1.65, linewidth=.5)
    ax.grid()

    # Plot the density from y=0 to y=2 for x = 0
    fig, ax = plt.subplots()
    logpdfs = np.exp([
        vs_y.logpdf(0, {1:y}, None, {0:0})
        for y in np.linspace(0,2,50)
    ])
    ax.plot(np.linspace(0, 2, 50), logpdfs)
