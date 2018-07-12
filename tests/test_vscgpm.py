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

import hacks
import pytest
if not pytest.config.getoption('--integration'):
    hacks.skip('specify --integration to run integration tests')

import importlib
import json

from collections import namedtuple

import numpy as np
import pytest

from cgpm.venturescript.vscgpm import VsCGpm


source_abstract = """
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

[define transition
  (lambda (N)
    (resimulation_mh default one N))]
"""


source_concrete = """
define make_cgpm = () -> {
    // Population variables.
    assume sigma = gamma(1, 1);

    // Latent variables.
    assume x = mem((rowid) ~> {
        normal(0, sigma)
    });

    assume simulate_m = mem((rowid, w) ~> {
        normal(x(rowid), sigma)
    });

    assume simulate_y = mem((rowid, w) ~> {
        uniform_continuous(w - 10, w + 10)
    });

    assume simulators = [simulate_m, simulate_y];
};

define observe_m = (rowid, w, value, label) -> {
    $label: observe simulate_m($rowid, $w) = value;
};

define observe_y = (rowid, w, value, label) -> {
    $label: observe simulate_y($rowid, $w) = value;
};

define observers = [observe_m, observe_y];

define inputs = ["w"];

define transition = (N) -> {
    resimulation_mh(default, one, N)
};
"""

Case = namedtuple('Case', ['source', 'mode'])
cases = [
    Case(source_abstract, 'church_prime'),
    Case(source_concrete, 'venture_script'),
]

@pytest.mark.parametrize('case', cases)
def test_wrong_outputs(case):
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1], inputs=[2], source=case.source, mode=case.mode)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2,3], inputs=[2], source=case.source, mode=case.mode)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2,2], inputs=[2], source=case.source, mode=case.mode)

@pytest.mark.parametrize('case', cases)
def test_wrong_inputs(case):
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[1], source=case.source, mode=case.mode)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[], source=case.source, mode=case.mode)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[3,4], source=case.source, mode=case.mode)

@pytest.mark.parametrize('case', cases)
def test_incorporate_unincorporate(case):
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=case.source, mode=case.mode)

    OBS = [[1.2, .2], [1, 4]]
    EV = [0, 2]

    rowid = 0

    # Missing evidence.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid, {0:OBS[rowid][0], 1:OBS[rowid][1]}, {})
    # No query.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid, {}, {3:EV[rowid]})

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

    # Test observation stable after transition.
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
    cgpm.transition(N=10)
    test_samples_match()

    # Test that simulating a hypothetical twice is different.
    first = cgpm.simulate(-100, [0, 1], None, {3:4})
    second = cgpm.simulate(-100, [0, 1], None, {3:4})
    assert first != second

    # Test observations resampled after transition.
    cgpm.unincorporate(1)
    with pytest.raises(ValueError):
        cgpm.simulate(1, [0,1])
    cgpm.transition(N=10)
    sample = cgpm.simulate(1, [0,1], None, {3:EV[rowid]})
    assert not np.allclose(sample[0], OBS[rowid][0])
    assert not np.allclose(sample[1], OBS[rowid][1])


@pytest.mark.parametrize('case', cases)
def test_serialize(case):
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=case.source, mode=case.mode)
    cgpm.incorporate(0, {0:1, 1:2}, {3:0})
    cgpm.incorporate(1, {1:15}, {3:10})
    cgpm.transition(N=2)

    binary = cgpm.to_metadata()
    modname, attrname = binary['factory']
    module = importlib.import_module(modname)
    builder = getattr(module, attrname)

    # Load binary from dictionary.
    cgpm2 = builder.from_metadata(binary)

    # Load binary from JSON.
    cgpm3 = builder.from_metadata(json.loads(json.dumps(binary)))

    print
    for cgpm_test in [cgpm2]:
        assert cgpm.outputs == cgpm_test.outputs
        assert cgpm.inputs == cgpm_test.inputs
        assert cgpm.source == cgpm_test.source
        assert cgpm.obs == cgpm_test.obs

        sample = cgpm_test.simulate(0, [0,1])
        assert sample[0] == 1
        assert sample[1] == 2

        sample = cgpm_test.simulate(1, [1])
        assert sample[1] == 15

        cgpm_test.incorporate(1, {0:10})


@pytest.mark.xfail(strict=True, reason='Github issue #215 (serialization).')
def test_engine_composition():
    from cgpm.crosscat.engine import Engine

    X = np.asarray([[1, 2, 0, 1], [1, 1, 0, 0],])
    engine = Engine(X[:,[3]], outputs=[3], cctypes=['normal'], num_states=2)
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=source_abstract,)

    for i, row in enumerate(X):
        cgpm.incorporate(i, {0: row[0], 1: row[1]}, {3: row[3]})

    cgpm.transition(N=2)
    engine.compose_cgpm([cgpm, cgpm], multiprocess=True)
