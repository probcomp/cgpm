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

import base64
from datetime import datetime

import venture.shortcuts as vs

from cgpm.cgpm import CGpm
from cgpm.utils import config as cu
from cgpm.utils import general as gu


class InlineVsCGpm(CGpm):
    """Inline Venturescript CGpm specified by a single expression."""

    def __init__(self, outputs, inputs, rng=None, expression=None, **kwargs):
        # Set the rng.
        self.rng = rng if rng is not None else gu.gen_rng(1)
        seed = self.rng.randint(1, 2**31 - 1)
        # Basic input and output checking.
        if len(outputs) != 1 :
            raise ValueError('InlineVsCgpm produces 1 output only.')
        if len(set(inputs)) != len(inputs):
            raise ValueError('Non unique inputs: %s' % inputs)
        if not all(o not in inputs for o in outputs):
            raise ValueError('Duplicates: %s, %s' % (inputs, outputs))
        if not all(i not in outputs for i in inputs):
            raise ValueError('Duplicates: %s, %s' % (inputs, outputs))
        # Retrieve the expression.
        if expression is None:
            raise ValueError('Missing expression: %s' % expression)
        # Retrieve the ripl.
        self.ripl = vs.make_lite_ripl(seed=seed)
        self.ripl.set_mode('church_prime')
        # Save the outputs.
        self.outputs = outputs
        # Check correct inputs against the expression.
        self._validate_expression(expression, inputs)
        # Store the inputs and expression.
        self.inputs = inputs
        self.expression = expression
        self.ripl.execute_program(self.expression)

    def incorporate(self, rowid, query, evidence=None):
        return

    def unincorporate(self, rowid):
        return

    def logpdf(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        assert set(evidence.keys()) == set(self.inputs)
        assert query.keys() == self.outputs
        self.ripl.assume('expr', self.expression, label='expr_assume')
        args = self._retrieve_args(evidence)
        # Retrieve the conditional density.
        logp = self.ripl.observe(
            '(expr %s)' % args, query[self.outputs[0]], label='expr_observe')
        # Forget the label.
        self.ripl.forget('expr_observe')
        self.ripl.forget('expr_assume')
        return logp[0]

    def simulate(self, rowid, query, evidence=None, N=None):
        print rowid, query, evidence
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        if evidence is None: evidence = {}
        assert set(evidence.keys()) == set(self.inputs)
        assert query == self.outputs
        self.ripl.assume('expr', self.expression, label='expr')
        args = self._retrieve_args(evidence)
        sample = self.ripl.sample('(expr %s)' % (args,))
        self.ripl.forget('expr')
        return {self.outputs[0]: sample}

    def logpdf_score(self):
        raise NotImplementedError

    def transition(self, program=None, N=None):
        return

    def _gen_label(self):
        return 't%s%s' % (
            self.rng.randint(1,100),
            datetime.now().strftime('%Y%m%d%H%M%S%f'))

    def _retrieve_args(self, evidence):
        return str.join(' ', [str(evidence[i]) for i in self.inputs])

    def _validate_expression(self, expression, inputs):
        # We are expecting an expression of the form (lambda (<args>) exp)
        # so remove the whitespace, split by left parens, check the first
        # token is spaces, the second token is "lambda", the third token is the
        # the arguments.
        expression = expression.replace('\n', ' ')
        tokens = expression.split('(')
        assert all(t in ['', ' '] for t in tokens[0])
        assert tokens[1].strip() == 'lambda'
        arguments = tokens[2]
        assert len([i for i in arguments if i ==')']) == 1
        arguments = arguments.replace(')', '')
        arguments = arguments.split()
        assert len(arguments) == len(inputs)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['expression'] = self.expression
        metadata['factory'] = ('cgpm.venturescript.vsinline', 'InlineVsCGpm')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        cgpm = InlineVsCGpm(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            expression=metadata['expression'],
            rng=rng,)
        return cgpm
