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

from gpmcc.gpm import Gpm
from gpmcc.utils.general import gen_rng
from gpmcc.utils.general import log_pflip
from gpmcc.utils.general import logmeanexp
from gpmcc.utils.general import merge_dicts


class SyntheticXyGpm(Gpm):
    """Interface synthetic, two-dimensional GPMs that take arbitrary noise
    parameter in (0,1).

    Typically used for simulating and evaluating the log density of a zero
    correlation dataset, but any distribution over R2 is possible.
    """

    def __init__(self, outputs=None, inputs=None, noise=None, acc=1, rng=None):
        """Initialize the Gpm with given noise parameter.

        Parameters
        ----------
        noise : float
            Value in (0,1) indicating the noise level of the distribution.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        if rng is None:
            rng = gen_rng(0)
        if outputs is None:
            outputs = [0, 1]
        if noise is None:
            noise = .1
        self.rng = rng
        self.outputs = outputs
        self.noise = noise
        self.acc = acc

    def logpdf(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        _, wQE = self.weighted_samples(
            rowid, merge_dicts(query, evidence), self.acc)
        _, wE = self.weighted_samples(rowid, evidence, self.acc) if evidence\
            else (0, [0])
        return logmeanexp(wQE) - logmeanexp(wE)

    def simulate(self, rowid, query, evidence=None):
        if evidence is None: evidence = {}
        samples, weights = self.weighted_samples(rowid, evidence, self.acc)
        index = log_pflip(weights, rng=self.rng)
        return [samples[index][q] for q in query]

    def weighted_samples(self, rowid, evidence, size):
        """Ad-hoc importance network to use as test case for full network."""
        if not evidence:
            xs = [self.simulate_x(rowid) for _ in xrange(size)]
            ys = [self.simulate_yGx(rowid, x) for x in xs]
            samples = zip(xs, ys)
            weights = [0] * size
        elif evidence.keys() == self.outputs[:1]:
            x = evidence[self.outputs[0]]
            samples = [(x, self.simulate_yGx(rowid, x)) for _ in xrange(size)]
            weights = [self.logpdf_x(rowid, s[0]) for s in samples]
        elif evidence.keys() == self.outputs[1:]:
            y = evidence[self.outputs[1]]
            samples = [(self.simulate_x(rowid), y) for _ in xrange(size)]
            weights = [self.logpdf_yGx(rowid, y, s[0]) for s in samples]
        elif set(evidence.keys()) == set(self.outputs):
            x, y = evidence[self.outputs[0]], evidence[self.outputs[1]]
            samples = [[x,y]]*size
            weights = [self.logpdf_x(rowid,x)+self.logpdf_yGx(rowid,y,x)]*size
        else:
            raise ValueError('Bad arguments to weighted_samples.')
        return samples, weights

    def simulate_x(self, rowid):
        return self.x.simulate(rowid, [self.outputs[0]])

    def logpdf_x(self, rowid, x):
        return self.x.logpdf(rowid, {self.outputs[0]:x})

    def simulate_yGx(self, rowid, x):
        return self.y.simulate(rowid, [self.outputs[1]], {self.outputs[0]:x})

    def logpdf_yGx(self, rowid, y, x):
        return self.y.logpdf(rowid, {self.outputs[1]:y}, {self.outputs[0]:x})
