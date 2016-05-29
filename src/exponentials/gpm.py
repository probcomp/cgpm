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

class Gpm(object):
    """Interface for generative population models.

    Generative population models provide a procedural abstraction for
    multivariate probability densities and stochastic samplers.
    """

    def __init__(self, outputs, inputs, schema, rng):
        """Initialize the Gpm.

        Parameters
        ----------
        outputs : list<int>
            List of variables whose joint distribution is modeled by the GPM.
        inputs : list<int>, optional
            List of variables that must accompany any observation or query about
            a particular rowid.
        schema : **kwargs
            An opaque binary parsed by the GPM to initialize itself.
            Often contains information about hyperparameters, parameters,
            sufficient statistics, or metadata about the input variables.
        """

        raise NotImplementedError

    def incorporate(self, rowid, query, evidence):
        """Record an observed cell for the rowid member.

        rowid : token
            A unique token identifying the member.
        query : dict{int:value}
            The keys of `query` must be a subset of the `output` variables.
        evidence : dict{int:value}, optional
            Values of all `input` variables, if any.
        """
        raise NotImplementedError

    def unincorporate(self, rowid):
        """Remove all incorporated observations of `rowid`.

        An error will be thrown if the rowid was not previously incorporated.
        """
        raise NotImplementedError

    def logpdf(self, rowid, query, evidence):
        """Compute the conditional density of `query` given `evidence`.

        query : dict{int:value}
            The keys of `targets` must be a subet of the `output` variables.
        evidence : dict{int:value}, optional
            Values of all `input` variables, if any, as well as any partial
            observations of `output` variables (may not overlap with `query`).
        """
        raise NotImplementedError


    def simulate(self, rowid, query, evidence):
        """Produce a sample of the `query` variables conditioned on `evidence`.

        The sample must be drawn from the same density as `logpdf`.
        """
        raise NotImplementedError

    def logpdf_score(self):
        """Joint density of all observations and current latent state."""
        raise NotImplementedError

    def infer(self, program):
        """Apply an inference operator transitioning the internal state of GPM.

        program : **kwargs, dict
            Opaque binary parsed by the GPM to apply inference over its latents.
        """
        raise NotImplementedError
