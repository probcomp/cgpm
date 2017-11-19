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

"""Crash test for serialization of state and engine."""

import importlib
import json
import tempfile

import numpy as np

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.mixtures.view import View
from cgpm.regressions.forest import RandomForest
from cgpm.regressions.linreg import LinearRegression
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


def serialize_generic(Model, additional=None):
    """Model is either State or Engine class."""
    # Create categorical data of DATA_NUM_0 zeros and DATA_NUM_1 ones.
    data = np.random.normal(size=(100,5))
    data[:,0] = 0
    # Run a single chain for a few iterations.
    model = Model(
        data,
        cctypes=['bernoulli','normal','normal','normal','normal'],
        rng=gu.gen_rng(0))
    model.transition(N=1, checkpoint=1)
    # To metadata.
    metadata = model.to_metadata()
    modname = importlib.import_module(metadata['factory'][0])
    builder = getattr(modname, metadata['factory'][1])
    model = builder.from_metadata(metadata)
    # To JSON.
    json_metadata = json.dumps(model.to_metadata())
    model = builder.from_metadata(json.loads(json_metadata))
    # To pickle.
    with tempfile.NamedTemporaryFile(prefix='gpmcc-serialize') as temp:
        with open(temp.name, 'w') as f:
            model.to_pickle(f)
        with open(temp.name, 'r') as f:
            # Use the file itself
            model = Model.from_pickle(f, rng=gu.gen_rng(10))
            if additional:
                additional(model)
        # Use the filename as a string
        model = Model.from_pickle(temp.name, rng=gu.gen_rng(10))
        if additional:
            additional(model)


def test_state_serialize():
    serialize_generic(State)


def test_engine_serialize():
    def additional(engine):
        e = engine.to_metadata()
        # Only one dataset per engine, not once per state.
        assert 'X' in e
        assert 'X' not in e['states'][0]
        # Each state should be populated with dataset when retrieving.
        s = engine.get_state(0)
        assert 'X' in s.to_metadata()
    serialize_generic(Engine, additional=additional)


def test_view_serialize():
    data = np.random.normal(size=(100,5))
    data[:,0] = 0
    # Run a single chain for a few iterations.
    outputs = [2,4,6,8,10]
    X = {c:data[:,i].tolist() for i,c in enumerate(outputs)}
    model = View(
        X,
        cctypes=['bernoulli','normal','normal','normal','normal'],
        outputs=[1000]+outputs,
        rng=gu.gen_rng(0))
    model.transition(N=1)
    # Pick out some data.
    # To metadata.
    metadata = model.to_metadata()
    modname = importlib.import_module(metadata['factory'][0])
    builder = getattr(modname, metadata['factory'][1])
    model2 = builder.from_metadata(metadata)
    # Pick out some data.
    assert np.allclose(model.alpha(), model.alpha())
    assert dict(model2.Zr()) == dict(model.Zr())
    assert np.allclose(
        model.logpdf(-1, {0:0, 1:1}, {2:0}),
        model2.logpdf(-1, {0:0, 1:1}, {2:0}))
    assert np.allclose(
        model.logpdf(-1, {0:0, 1:1}),
        model2.logpdf(-1, {0:0, 1:1}))


def test_serialize_composite_cgpm():
    rng = gu.gen_rng(2)

    # Generate the data.
    cctypes, distargs = cu.parse_distargs([
        'categorical(k=3)',     # RandomForest          0
        'normal',               # LinearRegression      1
        'categorical(k=3)',     # GPMCC                 2
        'poisson',              # GPMCC                 3
        'normal',               # GPMCC                 4
        'lognormal'             # GPMCC                 5
        ])
    T, Zv, Zc = tu.gen_data_table(
        35, [.4, .6], [[.33, .33, .34], [.5, .5]],
        cctypes, distargs, [.2]*len(cctypes), rng=rng)
    D = np.transpose(T)

    # Create GPMCC.
    state = State(
        D[:,2:], outputs=[2,3,4,5], cctypes=cctypes[2:],
        distargs=distargs[2:], rng=rng)

    # Create a Forest.
    forest = RandomForest(
        outputs=[0],
        inputs=[1,2,3,4],
        distargs={
            'inputs': {
                'stattypes': [cctypes[i] for i in [1,2,3,4]],
                'statargs': [distargs[i] for i in [1,2,3,4]]},
            'k': distargs[0]['k']},
        rng=rng)

    # Create a Regression.
    linreg = LinearRegression(
        outputs=[1],
        inputs=[3,4,5],
        distargs={
            'inputs': {
                'stattypes': [cctypes[i] for i in [3,4,5]],
                'statargs': [distargs[i] for i in [3,4,5]]}},
        rng=rng)

    # Incorporate the data.
    def incorporate_data(cgpm, rowid, row):
        cgpm.incorporate(
            rowid,
            {i: row[i] for i in cgpm.outputs},
            {i: row[i] for i in cgpm.inputs},
        )
    for rowid, row in enumerate(D):
        incorporate_data(forest, rowid, row)
        incorporate_data(linreg, rowid, row)

    # Compose the CGPMs.

    # Run state transitions.
    state.transition(N=10, progress=False)
    # Compose CGPMs, instructing State to run the transitions.
    token_forest = state.compose_cgpm(forest)
    token_linreg = state.compose_cgpm(linreg)
    state.transition_foreign(N=10, cols=[forest.outputs[0], linreg.outputs[0]])

    # Now run the serialization.
    metadata = state.to_metadata()
    state2 = State.from_metadata(metadata)

    # Check that the tokens are in state2.
    assert token_forest in state2.hooked_cgpms
    assert token_linreg in state2.hooked_cgpms

    # The hooked cgpms must be unique objects after serialize/deserialize.
    assert state.hooked_cgpms[token_forest] != state2.hooked_cgpms[token_forest]
    assert state.hooked_cgpms[token_linreg] != state2.hooked_cgpms[token_linreg]

    # Check that the log scores of the hooked cgpms agree.
    assert np.allclose(
        state.hooked_cgpms[token_forest].logpdf_score(),
        state2.hooked_cgpms[token_forest].logpdf_score())
    assert np.allclose(
        state.hooked_cgpms[token_linreg].logpdf_score(),
        state2.hooked_cgpms[token_linreg].logpdf_score())

    # Now run some tests for the engine.
    e = Engine(
        D[:,2:], outputs=[2,3,4,5], cctypes=cctypes[2:],
        distargs=distargs[2:], num_states=2, rng=rng)
    e.compose_cgpm([forest, forest], multiprocess=1)
    e.compose_cgpm([linreg, linreg], multiprocess=1)
    e.transition_foreign(N=1, cols=[forest.outputs[0], linreg.outputs[0]])
    e.dependence_probability(0,1)
    e.simulate(-1, [0,1], {2:1}, multiprocess=0)
    e.logpdf(-1, {1:1}, {2:1, 0:0}, multiprocess=0)

    state3 = e.get_state(0)

    # There is no guarantee that the logpdf score improves with inference, but
    # it should reduce by more than a few nats.
    def check_logpdf_delta(before, after):
        return before < after or (after-before) < 5
    check_logpdf_delta(
        before=state.hooked_cgpms[token_forest].logpdf_score(),
        after=state3.hooked_cgpms[token_forest].logpdf_score())
    check_logpdf_delta(
        before=state.hooked_cgpms[token_linreg].logpdf_score(),
        after=state3.hooked_cgpms[token_linreg].logpdf_score())
