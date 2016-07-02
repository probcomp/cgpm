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


import os
import pytest

from cgpm.utils.config import check_env_debug

token = 'GPMCCDEBUG'

def test_debug_none():
    if token in os.environ:
        del os.environ[token]
    assert not check_env_debug()

def test_debug_false():
    os.environ[token] = '0'
    assert not check_env_debug()

def test_debug_true():
    os.environ[token] = '1'
    assert check_env_debug()
