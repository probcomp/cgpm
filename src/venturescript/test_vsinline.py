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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cgpm.venturescript.vsinline import InlineVsCGpm

def test_input_matches_args():
    InlineVsCGpm([0], [], expression='(lambda () (normal 0 1))')
    InlineVsCGpm([0], [], expression='(lambda() (normal 0 1))')
    InlineVsCGpm([0], [], expression='(lambda(  ) (normal 0 1))')
    InlineVsCGpm([0], [], expression='(lambda (  ) (normal 0 1))')
    InlineVsCGpm([0], [1], expression='(lambda (a) (normal 0 1))')
    InlineVsCGpm([0], [1], expression='(lambda(a ) (normal 0 1))')
    InlineVsCGpm([0], [1], expression='(lambda( a ) (normal 0 1))')
    InlineVsCGpm([0], [1], expression='(lambda( ab) (normal 0 1))')
    InlineVsCGpm([0], [1,2], expression='(lambda (a b) (normal 0 1))')
    InlineVsCGpm([0], [1,2], expression='(lambda(a b) (normal 0 1))')
    InlineVsCGpm([0], [1,2], expression='(lambda(a b ) (normal 0 1))')
    InlineVsCGpm([0], [1,2], expression='(lambda (a b ) (normal 0 1))')

    # This is the test case for the figure.
    InlineVsCGpm([0], [],
        expression='(lambda () (uniform_continuous -4.71 4.71))')
    InlineVsCGpm([0], [1],
        expression='''
            (lambda (x)
                (if (> (cos x) 0)
                    (uniform_continuous (- (cos x) .5) (cos x))
                    (uniform_continuous (cos x) (+ (cos x) .5))))
        ''')

    with pytest.raises(Exception):
        InlineVsCGpm([0], [], expression='(lambda (a) (normal 0 1))')
    with pytest.raises(Exception):
        InlineVsCGpm([0], [1,2], expression='(lambda (a) (normal 0 1))')


def test_simulate_uniform():
    vs = InlineVsCGpm([0], [],
        expression='(lambda () (uniform_continuous -4.71 4.71))')

    lp = vs.logpdf(0, {0:0})
    for x in np.linspace(-4.70, 4.70, 100):
        assert np.allclose(vs.logpdf(0, {0:x}), lp)
    assert np.isinf(vs.logpdf(0, {0:12}))

    samples = vs.simulate(0, [0], evidence=None, N=200)
    extracted = [s[0] for s in samples]
    fig, ax = plt.subplots()
    ax.hist(extracted)

def test_simulate_noisy_cos():
    vs_x = InlineVsCGpm([0], [],
        expression='(lambda () (uniform_continuous -4.71 4.71))')

    vs_y = InlineVsCGpm([1], [0],
        expression='(lambda (x) (normal (cos x) 0.25))')

    # XXX TODO: Debug why this versions returns a noiseless cosine.
    InlineVsCGpm([1], [0],
        expression='''
            (lambda (x)
                (if (> (cos x) 0)
                    (uniform_continuous (- (cos x) .5) (cos x))
                    (uniform_continuous (cos x) (+ (cos x) .5))))
        ''')

    samples_x = vs_x.simulate(0, [0], evidence=None, N=200)
    samples_y = [vs_y.simulate(0, [1], sx) for sx in samples_x]

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
    logpdfs = np.exp(
        [vs_y.logpdf(0, {1:y}, {0:0}) for y in np.linspace(0,2,50)])
    ax.plot(np.linspace(0, 2, 50), logpdfs)
