# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015, Google, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Minimal test suite targeting cgpm.crosscat.loomcat.
"""

import hacks
import pytest
if not pytest.config.getoption('--integration'):
    hacks.skip('specify --integration to run integration tests')

import importlib
import itertools
import json

import numpy

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.utils import config as cu
from cgpm.utils import general as gu
from cgpm.utils import test as tu


def test_errors():
    """Targets loomcat._validate_transition."""
    D, Zv, Zc = tu.gen_data_table(
        n_rows=150,
        view_weights=None,
        cluster_weights=[[.2,.2,.2,.4], [.3,.2,.5],],
        cctypes=['normal']*6,
        distargs=[None]*6,
        separation=[0.95]*6,
        view_partition=[0,0,0,1,1,1],
        rng=gu.gen_rng(12)
    )

    state = State(
        D.T,
        outputs=range(10, 16),
        cctypes=['normal']*len(D),
        distargs=[None]*6,
        rng=gu.gen_rng(122),
    )

    engine = Engine(
        D.T,
        outputs=range(10, 16),
        cctypes=['normal']*len(D),
        distargs=[None]*6,
        rng=gu.gen_rng(122),
    )

    def check_errors(cgpm):
        with pytest.raises(ValueError):
            cgpm.transition_loom(N=10, S=5)
        with pytest.raises(ValueError):
            cgpm.transition_loom(N=10, kernels=['alpha'])
        with pytest.raises(ValueError):
            cgpm.transition_loom(N=10, progress=True)
        with pytest.raises(ValueError):
            cgpm.transition_loom(N=10, progress=True)
        with pytest.raises(ValueError):
            cgpm.transition_loom(N=10, checkpoint=2)
        cgpm.transition_loom(N=2)

    check_errors(state)
    check_errors(engine)


def test_multiple_stattypes():
    '''Test cgpm statistical types are heuristically converted to Loom types.'''
    cctypes, distargs = cu.parse_distargs([
        'normal',
        'poisson',
        'bernoulli',
        'categorical(k=4)',
        'lognormal',
        'exponential',
        'beta',
        'geometric',
        'vonmises'
    ])

    T, Zv, Zc = tu.gen_data_table(
        200,
        [1],
        [[.25, .25, .5]],
        cctypes,
        distargs,
        [.95]*len(cctypes),
        rng=gu.gen_rng(10)
    )

    engine = Engine(
        T.T,
        cctypes=cctypes,
        distargs=distargs,
        rng=gu.gen_rng(15),
        num_states=16,
    )

    logscore0 = engine.logpdf_score()
    engine.transition_loom(N=5)
    logscore1 = engine.logpdf_score()
    assert numpy.mean(logscore1) > numpy.mean(logscore0)

    # Check serializeation.
    metadata = engine.to_metadata()
    modname = importlib.import_module(metadata['factory'][0])
    builder = getattr(modname, metadata['factory'][1])
    engine2 = builder.from_metadata(metadata)

    # To JSON.
    json_metadata = json.dumps(engine.to_metadata())
    engine3 = builder.from_metadata(json.loads(json_metadata))

    # Assert all states in engine, engine2, and engine3 have same loom_path.
    loom_paths = list(itertools.chain.from_iterable(
        [s._loom_path for s in e.states]
        for e in [engine, engine2, engine3]
    ))
    assert all(p == loom_paths[0] for p in loom_paths)

    engine2.transition(S=5)
    dependence_probability = engine2.dependence_probability_pairwise()

    assert numpy.all(dependence_probability > 0.85)


def test_dependence_probability():
    '''Test that Loom correctly recovers a 2-view dataset.'''
    D, Zv, Zc = tu.gen_data_table(
        n_rows=150,
        view_weights=None,
        cluster_weights=[[.2,.2,.2,.4], [.3,.2,.5],],
        cctypes=['normal']*6,
        distargs=[None]*6,
        separation=[0.95]*6,
        view_partition=[0,0,0,1,1,1],
        rng=gu.gen_rng(12)
    )

    engine = Engine(
        D.T,
        outputs=[7, 2, 12, 80, 129, 98],
        cctypes=['normal']*len(D),
        distargs=[None]*6,
        rng=gu.gen_rng(122),
        num_states=20,
    )

    logscore0 = engine.logpdf_score()
    engine.transition_loom(N=100)
    logscore1 = engine.logpdf_score()
    assert numpy.mean(logscore1) > numpy.mean(logscore0)

    dependence_probability = numpy.mean(
        engine.dependence_probability_pairwise(),
        axis=0)

    assert dependence_probability[0,1] > 0.8
    assert dependence_probability[1,2] > 0.8
    assert dependence_probability[0,2] > 0.8

    assert dependence_probability[3,4] > 0.8
    assert dependence_probability[4,5] > 0.8
    assert dependence_probability[3,5] > 0.8

    assert dependence_probability[0,3] < 0.2
    assert dependence_probability[0,4] < 0.2
    assert dependence_probability[0,5] < 0.2

    assert dependence_probability[1,3] < 0.2
    assert dependence_probability[1,4] < 0.2
    assert dependence_probability[1,5] < 0.2

    assert dependence_probability[2,3] < 0.2
    assert dependence_probability[2,4] < 0.2
    assert dependence_probability[2,5] < 0.2
