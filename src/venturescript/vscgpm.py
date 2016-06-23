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

        # Evidence and labels for incorporate/unincorporate.
        self.obs = defaultdict(lambda: defaultdict(dict))


    def incorporate(self, rowid, query, evidence=None):
        evidence = self._validate_incorporate(rowid, query, evidence)
        for q, value in query.iteritems():
            self._observe_cell(rowid, q, value, evidence)

    def unincorporate(self, rowid):
        if rowid not in self.obs:
            raise ValueError('Never incorporated: %d' % rowid)
        for q in self.outputs:
            self._forget_cell(rowid, q)
        assert len(self.obs[rowid]['labels']) == 0
        del self.obs[rowid]

    def logpdf(self, rowid, query, evidence=None):
        raise NotImplementedError

    def simulate(self, rowid, query, evidence=None, N=None):
        pass

    def logpdf_score(self):
        raise NotImplementedError

    def transition(self, program=None, steps=None):
        if program is None:
            if steps is None: steps = 1000
            program = '[infer (mh default one %s)]' % steps
        self.ripl.execute_program(program)

    # --------------------------------------------------------------------------
    # Internal helpers.

    def _observe_cell(self, rowid, query, value, evidence=None):
        if evidence is None: evidence = {}
        inputs = [evidence[i] for i in self.inputs]
        label = '\'t'+cu.timestamp().replace('-','')
        args = str.join(' ', map(str, [rowid] + inputs + [value, label]))
        i = self.outputs.index(query)
        self.ripl.evaluate('((lookup observers %i) %s)' % (i, args))
        self.obs[rowid]['labels'][query] = label[1:]

    def _forget_cell(self, rowid, query):
        if query not in self.obs[rowid]['labels']: return
        label = self.obs[rowid]['labels'][query]
        self.ripl.forget(label)
        del self.obs[rowid]['labels'][query]

    def _validate_incorporate(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        # All evidence present, and no nan values.
        if set(evidence) != set(self.inputs):
            raise ValueError('Miss evidence: %s, %s' % (evidence, self.inputs))
        if not set.issubset(set(query), set(self.outputs)):
            raise ValueError('Miss query: %s,%s' % (query, self.outputs))
        if any(math.isnan(evidence[i]) for i in evidence):
            raise ValueError('Nan evidence: %s' % evidence)
        if any(math.isnan(query[i]) for i in query):
            raise ValueError('Nan query (nan): %s' % query)
        # Evidence optional if (rowid,q) previously incorporated, or must match.
        if rowid in self.obs:
            if any(q in self.obs[rowid]['labels'] for q in query):
                raise ValueError('Observation exists: %d, %s' % (rowid, query))
            if evidence: self._check_matched_evidence(rowid, evidence)
        else:
            self.obs[rowid]['evidence'] = dict(evidence)
        return self.obs[rowid]['evidence']

    def _validate_simulate(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        if any(math.isnan(evidence[i]) for i in evidence):
            raise ValueError('Nan evidence: %s' % evidence)
        if any(i in self.inputs or i in self.outputs for i in evidence):
            raise ValueError('Unknown variables: %s' % evidence)
        ev_in = {q:v for q,v in evidence.iteritems() if q in self.inputs}
        ev_out = {q:v for q,v in evidence.iteritems() if q in self.outputs}
        if rowid in self.obs:
            if ev_out and any(q in self.obs[rowid]['labels'] for q in ev_out):
                raise ValueError('Observation exists: %d, %s' % (rowid, query))
            if ev_in:
                self._check_matched_evidence(rowid, ev_in)
            else:
                ev_in = self.obs[rowid]['evidence']
        assert set(ev_in) == set(self.inputs)
        return ev_in, ev_out

    def _check_matched_evidence(self, rowid, evidence):
        if evidence != self.obs[rowid]['evidence']:
            raise ValueError('Unmatched evidence: %d, %s, %s' %
                (rowid, evidence, self.obs[rowid]['evidence']))
