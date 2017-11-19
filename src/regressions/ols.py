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

import base64
import cPickle
import math

from collections import OrderedDict
from collections import namedtuple

import numpy as np

from sklearn.linear_model import LinearRegression

from cgpm.cgpm import CGpm
from cgpm.utils import config as cu
from cgpm.utils import data as du
from cgpm.utils import general as gu


Data = namedtuple('Data', ['x', 'Y'])


class OrdinaryLeastSquares(CGpm):
    """Ordinary least squares linear model."""

    def __init__(self, outputs, inputs, hypers=None, params=None, distargs=None,
            rng=None):
        if params is None:
            params = {}
        self.outputs = outputs
        self.inputs = inputs
        self.rng = gu.gen_rng() if rng is None else rng
        assert len(self.outputs) == 1
        assert len(self.inputs) >= 1
        assert self.outputs[0] not in self.inputs
        assert len(distargs['inputs']['stattypes']) == len(self.inputs)
        self.input_cctypes = distargs['inputs']['stattypes']
        self.input_ccargs = distargs['inputs']['statargs']
        # Determine number of covariates (with 1 bias term) and number of
        # categories for categorical covariates.
        p, counts = zip(*[
            self._predictor_count(cctype, ccarg) for cctype, ccarg
            in zip(self.input_cctypes, self.input_ccargs)])
        self.p = sum(p)+1
        self.inputs_discrete = {i:c for i, c in enumerate(counts) if c}
        # Dataset.
        self.N = 0
        self.data = Data(x=OrderedDict(), Y=OrderedDict())
        # Noise of the regression.
        self.noise = params.get('noise', 1)
        # Regressor.
        self.regressor = params.get('regressor', None)
        if self.regressor is None:
            self.regressor = LinearRegression()

    def incorporate(self, rowid, observation, inputs=None):
        assert rowid not in self.data.x
        assert rowid not in self.data.Y
        if self.outputs[0] not in observation:
            raise ValueError('No observation in incorporate: %s' % observation)
        x, y = self.preprocess(observation, inputs)
        self.N += 1
        self.data.x[rowid] = x
        self.data.Y[rowid] = y

    def unincorporate(self, rowid):
        try:
            del self.data.x[rowid]
            del self.data.Y[rowid]
        except KeyError:
            raise ValueError('No such observation: %d' % rowid)
        self.N -= 1

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert rowid not in self.data.x
        assert not constraints
        xt, yt = self.preprocess(targets, inputs)
        mean = self.regressor.predict([yt])[0]
        return logpdf_gaussian(xt, mean, self.noise)

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets == self.outputs
        assert not constraints
        if rowid in self.data.x:
            return {self.outputs[0]: self.data.x[rowid]}
        _xt, yt = self.preprocess(None, inputs)
        mean = self.regressor.predict([yt])[0]
        x = self.rng.normal(mean, self.noise, size=N)
        # XXX Temporarily disable noise in the OLS sample.
        sample = {self.outputs[0]: mean}
        return sample if not N else [sample] * N

    def logpdf_score(self):
        pass

    ##################
    # NON-GPM METHOD #
    ##################

    def transition(self, N=None):
        # Transition forest.
        if len(self.data.Y) > 0:
            self.regressor.fit(self.data.Y.values(), self.data.x.values())
            predictions = self.regressor.predict(self.data.Y.values())
            self.noise = \
                np.linalg.norm(self.data.x.values() - predictions)\
                / np.sqrt(self.N)

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        return

    def get_hypers(self):
        return {}

    def get_params(self):
        return {
            'noise': self.noise,
            'regressor': self.regressor,
        }

    def get_suffstats(self):
        return {}

    def get_distargs(self):
        return {
            'inputs': {
                'stattypes': self.input_cctypes,
                'statargs': self.input_ccargs,
            } ,
            'inputs_discrete': self.inputs_discrete,
            'p': self.p,
        }

    @staticmethod
    def name():
        return 'ordinary_least_squares'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return True

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################

    @staticmethod
    def _predictor_count(cct, cca):
        # XXX Determine statistical types and arguments of inputs.
        if cct == 'numerical' or cu.cctype_class(cct).is_numeric():
            p, counts = 1, None
        elif cca is not None and 'k' in cca:
            # In dummy coding, if the category has values {1,...,K} then its
            # code contains (K-1) entries, where all zeros indicates value K.
            p, counts = cca['k']-1, int(cca['k'])
        else:
            raise ValueError('Invalid stattype, stargs: %s, %s.' % (cct, cca))
        return int(p), counts

    def preprocess(self, targets, inputs):
        # Retrieve the value x of the target variable.
        if self.outputs[0] in inputs:
            raise ValueError('Cannot condition on output %s: %s'
                % (self.outputs, inputs.keys()))
        if targets:
            x = targets.get(self.outputs[0], None)
            if x is None or np.isnan(x):
                raise ValueError('Invalid targets: %s' % (targets,))
        else:
            x = None
        # Retrieve the input values and dummy code them.
        if set(inputs.keys()) != set(self.inputs):
            raise ValueError('OLS requires inputs %s: %s'
                % (self.inputs, inputs.keys()))
        y = [inputs[c] for c in self.inputs]
        if any(np.isnan(v) for v in y):
            raise ValueError('OLS cannot accept nan inputs: %s.' % (inputs,))
        y = du.dummy_code(y, self.inputs_discrete)
        assert len(y) == self.p - 1
        # Return target and inputs.
        return x, y

    ####################
    # SERLIAZE METHODS #
    ####################

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = {'x': self.data.x, 'Y': self.data.Y}
        metadata['distargs'] = self.get_distargs()
        metadata['params'] = self.get_params()
        metadata['factory'] = ('cgpm.regressions.ols', 'OrdinaryLeastSquares')

        # Pickle the sklearn regressor.
        regressor = metadata['params']['regressor']
        regressor_binary = base64.b64encode(cPickle.dumps(regressor))
        metadata['params']['regressor_binary'] = regressor_binary
        del metadata['params']['regressor']

        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None: rng = gu.gen_rng(0)
        # Unpickle the sklearn ols.
        skl_ols = cPickle.loads(
            base64.b64decode(metadata['params']['regressor_binary']))
        metadata['params']['regressor'] = skl_ols
        ols = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            params=metadata['params'],
            distargs=metadata['distargs'],
            rng=rng)
        # json keys are strings -- convert back to integers.
        x = ((int(k), v) for k, v in metadata['data']['x'].iteritems())
        Y = ((int(k), v) for k, v in metadata['data']['Y'].iteritems())
        ols.data = Data(x=OrderedDict(x), Y=OrderedDict(Y))
        ols.N = metadata['N']
        return ols

HALF_LOG2PI = 0.5 * math.log(2 * math.pi)
def logpdf_gaussian(x, mu, sigma):
    deviation = x - mu
    return - math.log(sigma) - HALF_LOG2PI \
        - (0.5 * deviation * deviation / (sigma * sigma))
