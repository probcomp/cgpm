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

from venture.exception import VentureException

from cgpm.venturescript.vscgpm import VsCGpm

source_abstract = """
[define make_cgpm (lambda ()
  (do
    ; Address space for inputs
    (assume inputs (dict (list "w" (dict))))

    ; Population variables.
    (assume sigma (gamma 1 1))

    ; Latent variables.
    (assume x
      (mem (lambda (rowid)
        (normal 0 sigma))))

    ; Output variables.
    (assume simulate_m
      (mem (lambda (rowid)
        (tag rowid 0 (normal (x rowid) sigma)))))

    (assume simulate_y
      (mem (lambda (rowid)
        (let ((w (lookup (lookup inputs "w") rowid)))
            (tag rowid 1 (uniform_continuous (- w 10) (+ w 10)))))))

    (assume outputs (list 'simulate_m 'simulate_y))))]

[define observe_m
  (lambda (rowid value label)
    (observe (simulate_m ,rowid) value ,label))]

[define observe_y
  (lambda (rowid value label)
    (observe (simulate_y ,rowid) value ,label))]

[define transition
  (lambda (N)
    (mh default one N))]

"""

source_concrete = """
define make_cgpm = () -> {
    // Address space for inputs.
    assume inputs = dict(["w", dict()]);

    // Population variables.
    assume sigma = gamma(1, 1);

    // Latent variables.
    assume x = mem((rowid) ~> {
        normal(0, sigma)
    });

    assume simulate_m = mem((rowid) ~> {
        normal(x(rowid), sigma)
    });

    assume simulate_y = mem((rowid) ~> {
        w = inputs["w"][rowid];
        uniform_continuous(w - 10, w + 10)
    });

    assume outputs = [[|simulate_m|], [|simulate_y|]];
};

define observe_m = (rowid, value, label) -> {
    $label: observe simulate_m($rowid) = value;
};

define observe_y = (rowid, value, label) -> {
    $label: observe simulate_y($rowid) = value;
};

define transition = (N) -> {
    mh(default, one, N)
};

"""

# Define source with client overriding observers.
source_abstract_observers_good = source_abstract + \
    '[define observers (list observe_m observe_y)]\n'
source_abstract_observers_bad = source_abstract + \
    '[define observers (list observe_m observe_y 2)]\n'

source_concrete_observers_good = source_concrete + \
    'define observers = [observe_m, observe_y];\n'
source_concrete_observers_bad = source_concrete + \
    'define observers = [observe_m, observe_y, 2];\n'

# Define test cases.
Case = namedtuple('Case', ['source', 'mode'])
cases = [
    Case(source_abstract,                   'church_prime'),
    Case(source_concrete,                   'venture_script'),
    Case(source_abstract_observers_good,    'church_prime'),
    Case(source_concrete_observers_good,    'venture_script'),
]

CaseObs = namedtuple('Case', ['source', 'obsok', 'mode'])
casesObs = [
    CaseObs(source_abstract,                   True, 'church_prime'),
    CaseObs(source_concrete,                   True, 'venture_script'),
    CaseObs(source_abstract_observers_good,    True, 'church_prime'),
    CaseObs(source_concrete_observers_good,    True, 'venture_script'),
    CaseObs(source_abstract_observers_bad,     False, 'church_prime'),
    CaseObs(source_concrete_observers_bad,     False, 'venture_script'),
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

@pytest.mark.parametrize('case', casesObs)
def test_wrong_observers(case):
    try:
        VsCGpm(outputs=[0,1], inputs=[2], source=case.source, mode=case.mode)
        assert case.obsok
    except ValueError:
        assert not case.obsok

@pytest.mark.parametrize('case', cases)
def test_incorporate_unincorporate(case):
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=case.source, mode=case.mode)

    observations = [[1.2, .2], [1, 4]]
    inputs = [0, 2]
    rowid0 = 0
    rowid1 = 1
    rowid2 = 2

    # Missing input will raise a lookup error in Venture.
    with pytest.raises(VentureException):
        cgpm.incorporate(rowid0 , {1: observations[rowid0][1]}, {})
    # No query.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid0, {}, {3: inputs[rowid0]})
    cgpm.incorporate(rowid0, {0: observations[rowid0][0]}, {3: inputs[rowid0]})
    # Duplicate observation.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid0, {0: observations[rowid0][0]})
    # Incompatible evidence.
    with pytest.raises(ValueError):
        cgpm.incorporate(rowid0, {1: observations[rowid0][1]},
            {3: inputs[rowid0]+1})

    cgpm.incorporate(rowid0, {1: observations[rowid0][1]}, {3: inputs[rowid0]})

    cgpm.incorporate(rowid1, {0: observations[rowid1][0]})
    cgpm.incorporate(rowid1, {1: observations[rowid1][1]}, {3: inputs[rowid1]})

    # Test observation stable after transition.
    def test_samples_match():
        # Test rowid0.
        sample = cgpm.simulate(rowid0, [0,1])
        assert sample[0] == observations[rowid0][0]
        assert sample[1] == observations[rowid0][1]
        # Test rowid1.
        sample = cgpm.simulate(rowid1, [1])
        assert sample[1] == observations[rowid1][1]
        sample = cgpm.simulate(1, [0])
        assert sample[0] == observations[rowid1][0]

    test_samples_match()
    cgpm.transition(N=10)
    test_samples_match()

    # Test simulating hypothetical rowid twice gives different results.
    first = cgpm.simulate(-100, [0, 1], None, {3:4})
    second = cgpm.simulate(-100, [0, 1], None, {3:4})
    assert first != second

    # Test simulating hypothetical rowid with conditions.
    cgpm.simulate(-100, [0], {1:1}, {3:4})

    # Test can unincorporate partial observation.
    cgpm.incorporate(rowid2, {0: 1})
    cgpm.unincorporate(rowid2)

    # Test observations resampled after transition.
    cgpm.unincorporate(1)
    cgpm.simulate(1, [0])
    with pytest.raises(VentureException):
        # Missing inputs, w is required for output 1.
        cgpm.simulate(1, [1])
    cgpm.transition(N=10)
    sample = cgpm.simulate(1, [0,1], None, {3: inputs[rowid1]})
    assert not np.allclose(sample[0], observations[rowid1][0])
    assert not np.allclose(sample[1], observations[rowid1][1])


@pytest.mark.parametrize('case', cases[:1])
def test_logpdf(case):
    cgpm = VsCGpm(outputs=[0,1], inputs=[3], source=case.source, mode=case.mode)
    # Test univariate logpdf.
    cgpm.logpdf(-1, {0:1}, None, None)
    # Test conditional logpdf.
    with pytest.raises(VentureException):
        # Conditioning on {1:1} requires value for input 3.
        cgpm.logpdf(-1, {0:1}, {1:1}, None)
    cgpm.logpdf(-1, {0:1}, {1:1}, {3:0})
    # Test joint logpdf. output 1 should be uniform on [-10,10] so 100 should
    # given infinite density. However, it appears that using observe to fix the
    # targets/constraints causes their log_joint to become 0 (seems that nobody
    # thought through the change of base measure that happens when conditioning
    # on a continuous value, but this is besides the point since the bug will
    # also manifest for discrete random variables). Therefore the solution is to
    # set_value_at/set_value_at2 to fix the targets/constraints.
    with pytest.raises(AssertionError):
        assert cgpm.logpdf(-1, {0:1, 1:100}, None, {3:0}) == float('-inf')

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

    for cgpm_test in [cgpm2]:
        assert cgpm.outputs == cgpm_test.outputs
        assert cgpm.inputs == cgpm_test.inputs
        assert cgpm.source == cgpm_test.source
        assert cgpm.labels == cgpm_test.labels

        sample = cgpm_test.simulate(0, [0,1])
        assert sample[0] == 1
        assert sample[1] == 2

        sample = cgpm_test.simulate(1, [1])
        assert sample[1] == 15

        assert cgpm_test._get_input_cell_value(0, 3) == 0
        assert cgpm_test._get_input_cell_value(1, 3) == 10

        assert cgpm_test._is_observed_output_cell(0, 0)
        assert cgpm_test._is_observed_output_cell(0, 1)
        assert cgpm_test._is_observed_output_cell(1, 1)
        assert not cgpm_test._is_observed_output_cell(1, 0)

        cgpm_test.incorporate(1, {0:10})
        assert cgpm_test._is_observed_output_cell(1, 0)


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
