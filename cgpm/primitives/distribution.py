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

from builtins import range
from cgpm.cgpm import CGpm
from cgpm.mixtures.dim import Dim
from cgpm.utils import general as gu


class DistributionGpm(CGpm):
    """Interface for generative population models representing univariate
    probability distribution.

    A typical DistributionGpm will have:
    - Sufficient statistics T, for the observed data X.
    - Parameters Q, for the likelihood p(X|Q).
    - Hyperparameters H, for the prior p(Q|H).

    Additionally, some DistributionGpms will require per query
    - Conditioning variables Y, for the distribution p(X|Q,H,Y=y).

    This interface is uniform for both collapsed and uncollapsed models.
    A collapsed model will typically have no parameters Q.
    An uncollapsed model will typically carry a single set of parameters Q,
    but may also carry an ensemble of parameters (Q1,...,Qn) for simple
    Monte Carlo averaging of queries. The collapsed case is "theoretically"
    recovered in the limit n \to \infty.
    """

    def __init__(self, outputs, inputs, hypers, params, distargs, rng):
        assert len(outputs) == 1
        assert not inputs
        self.outputs = list(outputs)
        self.inputs = []
        self.data = dict()
        self.rng = gu.gen_rng() if rng is None else rng

    def incorporate(self, rowid, observation, inputs=None):
        assert rowid not in self.data
        assert not inputs
        assert list(observation.keys()) == self.outputs

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert rowid not in self.data
        assert not inputs
        assert not constraints
        assert list(targets.keys()) == self.outputs

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert not constraints
        assert not inputs
        assert targets == self.outputs

    ##################
    # NON-GPM METHOD #
    ##################

    def transition_hypers(self, N=None):
        if N is None:
            N = 1
        dim = Dim(
            self.outputs, [-10**8]+self.inputs, cctype=self.name(),
            hypers=self.get_hypers(), distargs=self.get_distargs(),
            rng=self.rng)
        dim.clusters[0] = self
        dim.transition_hyper_grids(X=list(self.data.values()))
        for i in range(N):
            dim.transition_hypers()


    def transition_params(self):
        """Resample the parameters Q conditioned on all observations X
        from an approximate posterior P(Q|X,H)."""
        raise NotImplementedError

    def set_hypers(self, hypers):
        """Force the hyperparameters H to new values."""
        raise NotImplementedError

    def get_hypers(self):
        """Return a dictionary of hyperparameters."""
        raise NotImplementedError

    def get_params(self):
        """Return a dictionary of parameters."""
        raise NotImplementedError

    def get_suffstats(self):
        """Return a dictionary of sufficient statistics."""
        raise NotImplementedError

    def get_distargs(self):
        """Return a dictionary of distribution arguments."""
        raise NotImplementedError

    @staticmethod
    def construct_hyper_grids(X, n_grid=20):
        """Return a dict<str,list>, where grids['hyper'] is a list of
        grid points for the binned hyperparameter distribution.

        This method is included in the interface since each GPM knows the
        valid values of its hypers, and may also use data-dependent
        heuristics from X to create better grids.
        """
        raise NotImplementedError


    @staticmethod
    def name():
        """Return the name of the distribution as a string."""
        raise NotImplementedError

    @staticmethod
    def is_collapsed():
        """Is the sampler collapsed?"""
        raise NotImplementedError

    @staticmethod
    def is_continuous():
        """Is the pdf defined on a continuous set?"""
        raise NotImplementedError

    @staticmethod
    def is_conditional():
        """Does the sampler require conditioning variables Y=y?"""
        raise NotImplementedError

    @staticmethod
    def is_numeric():
        """Is the support of the pdf a numeric or a symbolic set?"""
        raise NotImplementedError
