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
from gpmcc.utils.general import merged


class DirectedXyGpm(Gpm):
    """Interface directed two-dimensional GPMs over the R2 plane."""

    def __init__(self, outputs=None, inputs=None, noise=None, acc=1, rng=None):
        """Initialize the Gpm with given noise parameter.

        Parameters
        ----------
        noise : float
            Value in (0,1) indicating the noise level of the distribution.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """
        if type(self) is DirectedXyGpm:
            raise Exception('Cannot directly insantiate DirectedXyGpm.')
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
        # Override the network in subclass.
        self.network = None

    def logpdf(self, rowid, query, evidence=None):
        if self.network is None:
            raise ValueError('self.network not defined by %s' % type(self))
        return self.network.logpdf(rowid, query, evidence)

    def simulate(self, rowid, query, evidence=None, N=None):
        if self.network is None:
            raise ValueError('self.network not defined by %s' % type(self))
        return self.network.simulate(rowid, query, evidence, N)

class UnDirectedXyGpm(Gpm):
    """Interface undirected two-dimensional GPMs over the R2 plane."""

    def __init__(self, outputs=None, inputs=None, noise=None, rng=None):
        if rng is None:
            rng = gen_rng(0)
        if outputs is None:
            outputs = [0, 1]
        if noise is None:
            noise = .1
        self.rng = rng
        self.outputs = outputs
        self.inputs = []
        self.noise = noise

    def logpdf(self, rowid, query, evidence):
        if not evidence:
            if len(query) == 2:
                x, y = query.values()
                return self.logpdf_joint(x, y)
            else:
                z = query.values()[0]
                return self.logpdf_maringal(z)
        else:
            assert len(evidence) == len(query) == 1
            z = evidence.values()[0]
            w = query.values()[0]
            return self.logpdf_conditional(w, z)

    def simulate(self, rowid, query, evidence=None, N=None):
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        if not evidence:
            sample = self.simulate_joint()
            return {q: sample[self.outputs.index(q)] for q in query}
        else:
            assert len(evidence) == len(query) == 1
            z = evidence.values()[0]
            return {query[0]: self.simulate_conditional(z)}

    # Internal simulators and assesors.

    def simulate_joint(self):
        raise NotImplementedError

    def simulate_conditional(self, z):
        raise NotImplementedError

    def logpdf_marginal(self, z):
        raise NotImplementedError

    def logpdf_joint(self, x, y):
        raise NotImplementedError

    def logpdf_conditional(self, w, z):
        raise NotImplementedError

    def mutual_information(self):
        raise NotImplementedError
