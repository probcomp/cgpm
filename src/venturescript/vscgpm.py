# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

from collections import defaultdict

import venture.lite.types as vt
import venture.shortcuts as vs
import venture.value.dicts as vd

from cgpm.cgpm import CGpm
from cgpm.utils import config as cu
from cgpm.utils import general as gu


class VsCGpm(CGpm):

    def __init__(self, outputs, inputs, ripl=None, source=None, rng=None):
        # Set the rng.
        self.rng = rng if rng is not None else gu.gen_rng(1)
        seed = self.rng.randint(1, 2**31 - 1)

        # Basic input and output checking.
        assert len(set(outputs)) == len(outputs)
        assert len(set(inputs)) == len(inputs)
        assert all(o not in inputs for o in outputs)
        assert all(i not in outputs for i in inputs)

        # Retrieve the ripl.
        self.ripl = ripl if ripl is not None else vs.make_lite_ripl(seed=seed)
        self.ripl.set_mode('church_prime')

        # Execute the program.
        if source is not None:
            self.ripl.execute_program_from_file(source)

        # Check correct outputs.
        assert len(outputs) == len(self.ripl.sample('simulators'))
        self.outputs = outputs

        # Check correct inputs.
        assert len(inputs) == self.ripl.evaluate('(size inputs)')
        self.inputs = inputs

        # Check overriden observers.
        assert len(self.outputs) == self.ripl.evaluate('(size observers)')

        # Labels for incorporate/unincorporate.
        self.labels = defaultdict(dict)

    def incorporate(self, rowid, query, evidence=None):
        # Some validation.
        if evidence is None: evidence = {}
        assert set(evidence.keys()) == set(self.inputs)
        assert not any(math.isnan(evidence[i]) for i in evidence)
        assert not any(math.isnan(query[i]) for i in query)

        inputs = [evidence[i] for i in self.inputs]

        for q, value in query.iteritems():
            label = '\'t'+cu.timestamp().replace('-','')
            args = str.join(' ', map(str, [rowid] + inputs + [value, label]))
            i = self.outputs.index(q)
            self.ripl.evaluate('((lookup observers %i) %s)' % (i, args))
            self.labels[rowid][q] = label

    def unincorporate(self, rowid):
        raise NotImplementedError

    def logpdf(self, rowid, query, evidence=None):
        raise NotImplementedError

    def simulate(self, rowid, query, evidence=None, N=None):
        raise NotImplementedError

    def logpdf_score(self):
        raise NotImplementedError

    def transition(self, **kwargs):
        raise NotImplementedError
