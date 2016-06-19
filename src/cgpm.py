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

class CGpm(object):
    """Interface for conditional generative population models.

    Conditional generative population models provide a procedural abstraction
    for multivariate probability densities and stochastic samplers.
    """

    def __init__(self, outputs, inputs, schema, rng):
        """Initialize the CGpm.

        Parameters
        ----------
        outputs : list<int>
            List of variables whose joint distribution is modeled by the GPM.
        inputs : list<int>, optional
            List of variables that must accompany any observation or query.
        schema : **kwargs
            An opaque binary parsed by the GPM to initialize itself.
            Often contains information about hyperparameters, parameters,
            sufficient statistics, configuration settings,
            or metadata about the input variables.
        rng : numpy.random.RandomState
            Source of entropy.
        """
        raise NotImplementedError

    def incorporate(self, rowid, query, evidence=None):
        """Record an observation for `rowid`.

        rowid : int
            A unique integer identifying the member.
        query : dict{int:value}
            The observed values. The keys of `query` must be a subset of the
            `output` variables, and `value` must be type-matched based on
            `schema`.
        evidence : dict{int:value}, optional
            Values of all required `input` variables for the `rowid`.
        """
        raise NotImplementedError

    def unincorporate(self, rowid):
        """Remove all incorporated observations of `rowid`."""
        raise NotImplementedError

    def logpdf(self, rowid, query, evidence=None):
        """Return the conditional density of `query` given `evidence`.

        query : dict{int:value}
            The keys of `query` must be a subset of the `output` variables.
        evidence : dict{int:value}, optional
            Values of all required `input` variables as well as any partial
            observations of `output` variables to condition on (may not overlap
            with `query`).
        """
        raise NotImplementedError

    def simulate(self, rowid, query, evidence=None, N=None):
        """Return N iid samples of `query` variables conditioned on `evidence`.

        The sample must be drawn from the same distribution whose density is
        assessed by `logpdf`.

        query : list<int>
            List of `output` variables to simulate.
        evidence : dict{int:value}, optional
            Values of all required `input` variables as well as any partial
            observations of `output` variables to condition on (may not overlap
            with `query`).
        N : int, optional
            Number of samples to return. If None, returns a single sample. If
            integer, results a list of samples of length N.
        """
        raise NotImplementedError

    def logpdf_score(self):
        """Return joint density of all observations and current latent state."""
        raise NotImplementedError

    def transition(self, **kwargs):
        """Apply an inference operator transitioning the internal state of GPM.

        program : keyword arguments
            Opaque binary parsed by the GPM to apply inference over its latents.
        """
        raise NotImplementedError
