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

import pytest

from cgpm.kde.mvkde import MultivariateKde
from cgpm.utils import general as gu
from cgpm.utils import test as gt

O   = 'outputs'
ST  = 'stattypes'
SA  = 'statargs'
N   = 'numerical'
C   = 'categorical'

def test_initialize():
    # Typical initialization.
    MultivariateKde(
        outputs=[0, 1], inputs=None,
        distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # No inputs allowed.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=[2],
            distargs={O: {ST:[N, C], SA: [{}, {'k': 2}]}})
    # At least one output.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[], inputs=[],
            distargs={O: {ST: [], SA:[]}})
    # Unique outputs.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 0], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure outputs in distargs.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs=None)
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={'output': {ST: [N, C], SA: [{}, {'k': 2}]}})
    # Ensure stattypes and statargs in distargs['outputs]'
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {'stattype': [N, C], SA :[{}, {'k': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], 'eland': [{}, {'k': 2}]}})
    # Ensure stattypes correct length.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C, N], SA: [{}, {'k': 2}]}})
    # Ensure statargs correct length.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, None, {'k': 2}]}})
    # Ensure number of categories provided as k.
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {'h': 2}]}})
    with pytest.raises(ValueError):
        MultivariateKde(
            outputs=[0, 1], inputs=None,
            distargs={O: {ST: [N, C], SA: [{}, {}]}})
