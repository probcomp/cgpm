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

import venture.shortcuts as vs
import venture.value.dicts as vd
import venture.lite.types as vt

from cgpm.cgpm import CGpm
from cgpm.utils import general as gu


class VsCGpm(CGpm):

    def __init__(self, outputs, inputs, ripl=None, source=None, rng=None):
        # Set the rng.
        self.rng = rng if rng is not None else gu.gen_rng(1)

        # Basic input and output checking.
        assert len(set(outputs)) == len(outputs)
        assert len(set(inputs)) == len(inputs)
        assert all(o not in inputs for o in outputs)
        assert all(i not in outputs for i in inputs)

        # Retrieve the ripl.
        self.ripl = ripl if ripl is not None else vs.make_lite_ripl()
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
        def check_observer_override(o):
            null = self.ripl.evaluate('(= (lookup observers %s) nil)' % o)
            return not null
        self.observers = {o for o in xrange(len(self.outputs))
            if check_observer_override(o)}

        # Labels for incorporate/unincorporate.
        self.observation_labels = dict()

    def incorporate(self, rowid, query, evidence=None):
        raise NotImplementedError

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
