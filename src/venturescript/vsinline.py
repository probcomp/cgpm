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
import string

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
        # Save the outputs.
        self.outputs = outputs
        # Check correct inputs against the expression.
        self._validate_expression_concrete(expression, inputs)
        # Store the inputs and expression.
        self.inputs = inputs
        self.expression = expression
        # Execute the program in the ripl to make sure it parses.
        self.ripl = vs.make_lite_ripl(seed=seed)
        self.ripl.execute_program(self.expression)
        self.ripl.execute_program('assume uniform = uniform_continuous')

    def incorporate(self, rowid, observation, inputs=None):
        return

    def unincorporate(self, rowid):
        return

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        if inputs is None:
            inputs = {}
        assert set(inputs.keys()) == set(self.inputs)
        assert targets.keys() == self.outputs
        assert not constraints
        self.ripl.assume('expr', self.expression, label='expr_assume')
        sp_args = self._retrieve_args(inputs)
        # Retrieve the conditional density.
        logp = self.ripl.observe(
            'expr (%s)' % (sp_args,),
            targets[self.outputs[0]],
            label='expr_observe',
        )
        # Forget the label.
        self.ripl.forget('expr_observe')
        self.ripl.forget('expr_assume')
        return logp[0]

    @gu.simulate_many
    def simulate(self, rowid, targets, constraints, inputs=None, N=None):
        if inputs is None:
            inputs = {}
        assert set(inputs.keys()) == set(self.inputs)
        assert targets == self.outputs
        assert not constraints
        self.ripl.assume('expr', self.expression, label='expr')
        sp_args = self._retrieve_args(inputs)
        sample = self.ripl.sample('expr(%s)' % (sp_args,))
        self.ripl.forget('expr')
        return {self.outputs[0]: sample}

    def logpdf_score(self):
        raise NotImplementedError

    def transition(self, program=None, N=None):
        return

    def _retrieve_args(self, inputs):
        return str.join(',', [str(inputs[i]) for i in self.inputs])

    def _validate_expression_abstract(self, expression, inputs):
        # We are expecting an expression of the form (lambda (<args>) (exp))
        # so remove the whitespace, split by left parens, check the first
        # token is spaces, the second token is "lambda", the third token is the
        # the arguments. Note this will fail if (exp) is just exp i.e. 1.
        expression = expression.replace('\n', ' ')
        tokens = expression.split('(')
        # assert all(t in ['', ' '] for t in tokens[0])
        assert tokens[1].strip() == 'lambda'
        arguments = tokens[2]
        assert len([i for i in arguments if i ==')']) == 1
        arguments = arguments.replace(')', '')
        arguments = arguments.split()
        assert len(arguments) == len(inputs)

    def _validate_expression_concrete(self, expression, inputs):
        # We are expecting an expression of the form (a,b,c) ~> {}
        # so remove the whitespace, split by left parens, check the first
        # token is spaces, the second token is "lambda", the third token is the
        # the arguments.
        # Eliminate surrounding whitespace.
        expression = expression.encode('ascii','ignore')
        expression = expression.translate(None, string.whitespace)
        # Retrieve symbols before ~>.
        tokens = expression.split('~>')
        # Eliminate the parens.
        arguments = tokens[0].replace('(','').replace(')','')
        arguments = [a for a in arguments.split(',') if a != '']
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
            rng=rng,
        )
        return cgpm
