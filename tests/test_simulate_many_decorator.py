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

from cgpm.utils import general as gu

class DummyCgpm():

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        return (self, rowid, targets, constraints, inputs, N)

def test_simulate_many_kwarg_none():
    """N is None as a kwarg."""
    gpm = DummyCgpm()
    assert gpm.simulate(None, [1,2]) \
        == (gpm, None, [1,2], None, None, None)
    assert gpm.simulate(None, [1,2], inputs={4:1}) \
        == (gpm, None, [1,2], None, {4:1}, None)
    assert  gpm.simulate(4, [1,2], {3:2}) \
        == (gpm, 4, [1,2], {3:2}, None, None)
    assert gpm.simulate(None, [1,2], {4:2}, inputs={5:2}) \
        == (gpm, None, [1,2], {4:2}, {5:2}, None)
    assert gpm.simulate(None, [1,2], {5:2}) \
        == (gpm, None, [1,2], {5:2}, None, None)
    assert gpm.simulate(2, [1,2], constraints={4:2}, inputs={5:2}) \
        == (gpm, 2, [1,2], {4:2}, {5:2}, None)

def test_simulate_many_kwarg_not_none():
    """N is not None, used as a named parameter."""
    gpm = DummyCgpm()
    assert gpm.simulate(None, [1,2], N=10) \
        == [(gpm, None, [1,2], None, None, 10)] * 10
    assert gpm.simulate(9, [1,2], inputs=None, N=7) \
        == [(gpm, 9, [1,2], None, None, 7)] * 7
    assert gpm.simulate(None, [1,2], {2:1}, N=10) \
        == [(gpm, None, [1,2], {2:1}, None, 10)] * 10
    assert gpm.simulate(None, [1,2], constraints={4:2}, inputs={5:2}, N=1) \
        == [(gpm, None, [1,2], {4:2}, {5:2}, 1)]

def test_simulate_many_positional():
    # N is positional.
    gpm = DummyCgpm()
    assert gpm.simulate(77, [1,2], None, None, 10) \
        == [(gpm, 77, [1,2], None, None, 10)] * 10
    assert gpm.simulate(None, [1,2], None, {4:1}, 7) \
        == [(gpm, None, [1,2], None, {4:1}, 7)] * 7
    assert gpm.simulate(None, [1,2], {5:2}, None, 1) \
        == [(gpm, None, [1,2], {5:2}, None, 1)]
    assert gpm.simulate(None, [1,2], None, {3:1}, None) \
        == (gpm, None, [1,2], None, {3:1}, None)
