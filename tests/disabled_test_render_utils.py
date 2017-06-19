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


test_dataset_dpmm = np.array([
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
])

test_dataset_with_distractors = np.array([
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
])

test_dataset_mixed = np.hstack((
    np.array(test_dataset_dpmm),
    RNG.normal(10, 5, size=[12, 6]),
))

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


# Global variables for test cases involving a CrossCat state.


VIEW, STATE = init_view_state(
    test_dataset_mixed_nan, 25, ['bernoulli']*6 + ['normal']*6)
ROW_NAMES = string_generator(13, 10)
COL_NAMES = string_generator(12, 7)


# Test cases


def test_viz_data():
    fig, _ax = ru.viz_data(test_dataset_mixed_nan,)
    fig.savefig(get_filename('test_viz_data.png'))

def test_viz_data_with_names():
    row_names = string_generator(12, 10)
    col_names = string_generator(6, 7)
    fig, _ax = ru.viz_data(
        test_dataset_dpmm, row_names=row_names, col_names=col_names)
    fig.savefig(get_filename('test_viz_data_with_names.png'))

def test_viz_wide_data():
    fig, _ax = ru.viz_data(test_dataset_wide)
    fig.savefig(get_filename('test_viz_wide_data.png'))

def test_viz_tall_data():
    fig, _ax = ru.viz_data(test_dataset_tall)
    fig.savefig(get_filename('test_viz_tall_data.png'))

def test_viz_view():
    fig, _ax = ru.viz_view(VIEW)
    fig.savefig(get_filename('test_viz_view.png'))

def test_viz_view_with_names():
    fig, _ax = ru.viz_view(VIEW, row_names=ROW_NAMES, col_names=COL_NAMES)
    fig.savefig(get_filename('test_viz_view_with_names.png'))

def test_viz_view_with_names_subsample():
    fig, _ax = ru.viz_view(
        VIEW,
        row_names=ROW_NAMES,
        col_names=COL_NAMES,
        subsample=2,
        seed=2,
        yticklabelsize='medium',
    )
    fig.savefig(get_filename('test_viz_view_with_names_subsample.png'))

def test_viz_state():
    fig, _ax = ru.viz_state(STATE)
    fig.savefig(get_filename('test_viz_state.png'))

def test_viz_state_with_names():
    fig, _ax = ru.viz_state(STATE, row_names=ROW_NAMES, col_names=COL_NAMES)
    fig.savefig(get_filename('test_viz_state_with_names.png'))

def test_viz_state_with_names_subsample():
    # The subsampled rows should be the same for all views since identical seed
    # is passed to viz_view_raw, and each function creates its own rng.
    fig, _ax = ru.viz_state(
        STATE,
        row_names=ROW_NAMES,
        col_names=COL_NAMES,
        subsample=4,
        seed=2,
        yticklabelsize='medium',
    )
    fig.savefig(get_filename('test_viz_state_with_names_subsample.png'))
