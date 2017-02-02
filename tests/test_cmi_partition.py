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
import pytest

from cgpm.crosscat.engine import DummyCgpm
from cgpm.crosscat.state import State
from cgpm.utils.general import gen_rng


def retrieve_state():
    X = np.eye(7)
    cctypes = ['normal'] * 7
    return State(
        X,
        outputs=[10,11,12,13,14,15,16],
        Zv={10:0, 11:0, 12:1, 13:2, 14:2, 15:2, 16:0},
        cctypes=cctypes,
        rng=gen_rng(2),
    )


def test_partition_mutual_information_query():
    state = retrieve_state()

    def check_expected_partitions(query, expected):
        blocks = state._partition_mutual_information_query(*query)
        assert len(blocks) == len(expected)
        for b in blocks:
            assert b in expected

    check_expected_partitions(
        query=([10], [11], {}),
        expected=[
            ([10], [11], {}),
    ])
    check_expected_partitions(
        query=([10,16], [11], {}),
        expected=[
            ([10,16], [11], {}),
    ])
    check_expected_partitions(
        query=([10,16], [11], {12:None}),
        expected=[
            ([10,16], [11], {}),
            ([], [], {12:None}),
    ])
    check_expected_partitions(
        query=([10,16], [11, 14], {12:None}),
        expected=[
            ([10,16], [11], {}),
            ([], [14], {}),
            ([], [], {12:None}),
    ])
    check_expected_partitions(
        query=([15, 16], [11, 14, 13], {12:None, 13:2, 10:-12}),
        expected=[
            ([15], [14,13], {13:2}),
            ([16], [11], {10:-12}),
            ([], [], {12:None}),
    ])
    check_expected_partitions(
        query=([15, 16], [11, 14, 13], {12:None, 13:2}),
        expected=[
            ([15], [14,13], {13:2}),
            ([16], [11], {}),
            ([], [], {12:None}),
    ])
    check_expected_partitions(
        query=([15, 16], [15, 16], {12:None, 13:2}),
        expected=[
            ([15], [15], {13:2}),
            ([16], [16], {}),
            ([], [], {12:None}),
    ])
    check_expected_partitions(
        query=([13, 14], [14, 13], {}),
        expected=[
            ([13, 14], [14,13], {}),
    ])

    # Connect variable 12 with variables in view 0.
    state.compose_cgpm(DummyCgpm(outputs=[100, 102], inputs=[10, 12]))

    check_expected_partitions(
        query=([15, 16], [11, 14, 13], {12:None, 13:2}),
        expected=[
            ([15], [14,13], {13:2}),
            ([16], [11], {12:None}),
    ])
    check_expected_partitions(
        query=([15, 16], [11, 14, 13], {12:None, 13:2}),
        expected=[
            ([15], [14,13], {13:2}),
            ([16], [11], {12:None}),
    ])
    check_expected_partitions(
        query=([15, 100, 16], [11, 14, 13], {102: -12, 12:None, 13:2}),
        expected=[
            ([15], [14,13], {13:2}),
            ([100, 16], [11], {12:None, 102: -12}),
    ])

    # Connect variables in view 0 with variables in view 2.
    state.compose_cgpm(DummyCgpm(outputs=[200, 202], inputs=[100, 12]))
    state.compose_cgpm(DummyCgpm(outputs=[300], inputs=[200, 13]))

    check_expected_partitions(
        query=([15, 16], [11, 14, 13], {12:None, 13:2, 10:-12}),
        expected=[
            ([15, 16], [11, 14, 13], {12:None, 13:2, 10:-12})
        ])
    check_expected_partitions(
        query=([15, 16], [11, 14, 13], {12:None, 13:2}),
        expected=[
            ([15, 16], [11, 14, 13], {12:None, 13:2})
    ])
    check_expected_partitions(
        query=([300], [202], {13:2}),
        expected=[
            ([300], [202], {13:2})
    ])
