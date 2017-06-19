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

"""Graphical test suite providing coverage for cgpm.utils.render."""

import os

from string import ascii_uppercase

import numpy as np

from cgpm.crosscat.state import State
from cgpm.mixtures.view import View

from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import render as ru


OUT = '/tmp/'
RNG = gu.gen_rng(7)
TIMESTAMP = cu.timestamp()


# Define datasets.


test_dataset_dpmm = [
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0],
]

test_dataset_with_distractors = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

X1 = np.array(test_dataset_dpmm)

X2 = RNG.normal(10, 5, size=[12, 6])

test_dataset_mixed = np.hstack((X1, X2))

test_dataset_mixed_nan = np.vstack((test_dataset_mixed, [np.nan]*12))

test_dataset_wide = np.hstack(
    (test_dataset_mixed, test_dataset_mixed, test_dataset_mixed))

test_dataset_tall = np.vstack(
    (test_dataset_mixed, test_dataset_mixed, test_dataset_mixed))


# Initialize DPMM and CrossCat models for the above data.


def init_view_state(data, iters, cctypes):
    if isinstance(data, list):
        data = np.array(data)
    D = len(data[0])
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(X, cctypes=cctypes, outputs=[1000] + outputs, rng=RNG)
    state = State(data[:, 0:D], outputs=outputs, cctypes=cctypes, rng=RNG)
    if iters > 0:
        view.transition(iters)
        state.transition(iters)
    return view, state

# Helpers

def string_generator(N=1, length=10):
    from random import choice
    return [
        (''.join(choice(ascii_uppercase) for _ in xrange(length)))
        for _ in xrange(N)
    ]

def get_filename(name):
    return os.path.join(OUT, '%s_%s' % (TIMESTAMP, name,))

# view1, state1 = init_binary_view_state(test_dataset_dpmm, 50)
# view2, state2 = init_binary_view_state(test_dataset_with_distractors, 50)

view3, state3 = init_view_state(
    test_dataset_mixed_nan, 25, ['bernoulli']*6 + ['normal']*6)

row_names_test = string_generator(12, 10)
col_names_test = string_generator(6, 7)
row_names3 = string_generator(13, 10)
col_names3 = string_generator(12, 7)


# Test cases


def test_viz_data():
    ru.viz_data(test_dataset_mixed_nan,
        savefile=get_filename('test_viz_data.png'))

def test_viz_data_with_names():
    ru.viz_data(
        test_dataset_dpmm, row_names=row_names_test, col_names=col_names_test,
        savefile=get_filename('test_viz_data_with_names.png'))

def test_viz_wide_data():
    ru.viz_data(
        test_dataset_wide, savefile=get_filename('test_viz_wide_data.png'))

def test_viz_tall_data():
    ru.viz_data(
        test_dataset_tall, savefile=get_filename('test_viz_tall_data.png'))

def test_viz_view():
    ru.viz_view(view3, savefile=get_filename('test_viz_view.png'))

def test_viz_view_with_names():
    ru.viz_view(
        view3, row_names=row_names3, col_names=col_names3,
        savefile=get_filename('test_viz_view_with_names.png'))

def test_viz_state():
    ru.viz_state(state3, savefile=get_filename('test_viz_state.png'))

def test_viz_state_with_names():
    ru.viz_state(
        state3, row_names=row_names3, col_names=col_names3,
        savefile=get_filename('test_viz_state_with_names.png'))
