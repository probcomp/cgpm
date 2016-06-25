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

import numpy as np

from cgpm.venturescript.vscgpm import VsCGpm


source = """
[define make_cgpm (lambda ()
  (do
    ; Population variables.
    (assume sigma (gamma 1 1))

    ; Latent variables.
    (assume x
      (mem (lambda (rowid)
        (normal 0 sigma))))

    ; Output variables.
    (assume simulate_m
      (mem (lambda (rowid w)
        (normal (x rowid) sigma))))

    (assume simulate_y
      (mem (lambda (rowid w)
        (uniform_continuous (- w 10) (+ w 10)))))

    (assume simulators (list simulate_m
                             simulate_y))))]

[define observe_m
  (lambda (rowid w value label)
    (observe (simulate_m ,rowid ,w) value ,label))]

[define observe_y
  (lambda (rowid w value label)
    (observe (simulate_y ,rowid ,w) value ,label))]

[define observers (list observe_m
                        observe_y)]

[define inputs (list 'w)]
"""


def test_wrong_outputs():
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1], inputs=[2], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2,3], inputs=[2], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2,2], inputs=[2], source=source)

def test_wrong_inputs():
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[1], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[3,4], source=source)


def test_incorporate_unincorporate():
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=source)

    OBS = [[1.2, .2], [1, 4]]
    EV = [0, 2]

    rowid = 0
    # Missing evidence.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid, {0:OBS[rowid][0], 1:OBS[rowid][1]}, {})
    cgpm.incorporate(rowid, {0:OBS[rowid][0]}, {3:EV[rowid]})
    # Duplicate observation.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid, {0:OBS[rowid][0]})
    # Incompatible evidence.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid, {1:OBS[rowid][1]}, {3:EV[rowid]+1})
    # Compatible evidence.
    cgpm.incorporate(rowid, {1:OBS[rowid][1]}, {3:EV[rowid]})

    rowid = 1
    cgpm.incorporate(rowid, {1:OBS[rowid][1]}, {3:EV[rowid]})
    # Optional evidence.
    cgpm.incorporate(rowid, {0:OBS[rowid][0]})

    # Test obsevation stable after transition.
    def test_samples_match():
        # Check all samples match.
        sample = cgpm.simulate(0, [0,1])
        assert sample[0] == OBS[0][0]
        assert sample[1] == OBS[0][1]

        sample = cgpm.simulate(1, [1])
        assert sample[1] == OBS[rowid][1]
        sample = cgpm.simulate(1, [0])
        assert sample[0] == OBS[rowid][0]

    test_samples_match()
    cgpm.transition(steps=10)
    test_samples_match()

    # Test observations resampled after transition.
    cgpm.unincorporate(1)
    with pytest.raises(ValueError):
        cgpm.simulate(1, [0,1])
    cgpm.transition(steps=10)
    sample = cgpm.simulate(1, [0,1], {3:EV[rowid]})
    assert not np.allclose(sample[0], OBS[rowid][0])
    assert not np.allclose(sample[1], OBS[rowid][1])


def test_serialize():
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=source)
    cgpm.incorporate(0, {0:1, 1:2}, {3:0})
    cgpm.incorporate(1, {1:15}, {3:10})
    cgpm.transition(steps=2)

    binary = cgpm.to_metadata()
    modname, attrname = binary['factory']
    module = importlib.import_module(modname)
    builder = getattr(module, attrname)
    cgpm2 = builder.from_metadata(binary)

    assert cgpm.outputs == cgpm2.outputs
    assert cgpm.inputs == cgpm2.inputs
    assert cgpm.source == cgpm2.source

    sample = cgpm2.simulate(0, [0,1])
    assert sample[0] == 1
    assert sample[1] == 2

    sample = cgpm2.simulate(1, [1])
    assert sample[1] == 15

    cgpm2.incorporate(1, {0:10})
