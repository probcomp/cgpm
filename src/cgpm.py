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
    """Interface for composable generative population models.

    Composable generative population models provide a computational abstraction
    for multivariate probability densities and stochastic samplers.
    """

    def __init__(self, outputs, inputs, schema, rng):
        """Initialize the CGpm.

        Parameters
        ----------
        outputs : list<int>
            List of variables whose joint distribution is modeled by the CGpm.
            The CGpm is required to simulate and evaluate the log density
            of an arbitrary susbet of output variables, by marginalizing and/or
            conditioning on another (disjoint) subset of output variables.
        inputs : list<int>, optional
            List of variables unmodeled by the CGpm which are needed
            on a per-row basis. An example would be the covariate variables
            for a single output linear regression CGpm.
        schema : **kwargs
            An opaque binary parsed by the CGpm to initialize itself.
            Often contains information about hyperparameters, parameters,
            sufficient statistics, configuration settings,
            or metadata about the input variables.
        rng : numpy.random.RandomState
            Source of entropy.
        """
        raise NotImplementedError

    def incorporate(self, rowid, query, evidence=None):
        """Record an observation for `rowid` into the dataset.

        rowid : int
            A unique integer identifying the member.
        query : dict{int:value}
            The observed values. The keys of `query` must be a subset of the
            `output` variables, and `value` must be type-matched based on
            the statistical data type of that variable. Missing values may
            be either omitted, or specified as float(nan).
        evidence : dict{int:value}, optional
            Values of all required `input` variables for the `rowid`.
        """
        raise NotImplementedError

    def unincorporate(self, rowid):
        """Remove all incorporated observations of `rowid` from the dataset."""
        raise NotImplementedError

    def logpdf(self, rowid, query, evidence=None):
        """Return the conditional density of `query` given `evidence`.

        rowid : int, (use None to indicate a hypothetical row)
            Specifies the identity of the member against which to evaluate
            the log density. Specifying an existing `rowid` automatically
            populates all incorporated output variables for that rowid the
            `evidence`, and typically conditions on any current unexposed latent
            state for that `rowid`. If the `rowid` is hypothetical, then
            unexposed latent state is usually marginalized over or sampled.

        query : dict{int:value}
            The keys of `query` must be a subset of the `output` variables.
            If `rowid` corresponds to an existing member, it is an error
            to query any output variable for that `rowid` which has already
            been incorporated.

        evidence : dict{int:value}, optional
            The keys of `evidence` must contain all `input` variables, as well
            as an optional collection of partial observations of `output`
            variables to condition on. No keys in the `evidence` can overlap
            with keys in `query`. If `rowid` identifies a previously
            incorporated member, then all data for the `rowid` will be
            automatically used as `evidence`, and therefore the only valid keys
            in `evidence` are those `output` variables for which no observation
            was incorporated.
        """
        raise NotImplementedError

    def simulate(self, rowid, query, evidence=None, N=None):
        """Return N iid samples of `query` variables conditioned on `evidence`.

        The sample must be drawn from the same distribution whose density is
        assessed by `logpdf`.

        rowid : int, (use None to indicate a hypothetical row)
            Specifies the identity of the member to simulate from.
            Specifying an existing `rowid` automatically populates all
            incorporated output variables for that rowid the `evidence`, and
            typically conditions on any current unexposed latent state for that
            `rowid`. If the `rowid` is hypothetical, then unexposed latent state
            is usually marginalized over or sampled.

        query : list<int>
            List of `output` variables to simulate. If `rowid` corresponds to
            an existing member, it is an error to simulate any output variable
            which has already been incorporated.

        evidence : dict{int:value}, optional
            The keys of `evidence` must contain all `input` variables, as well
            as an optional collection of partial observations of `output`
            variables to condition on. No keys in the `evidence` can overlap
            with keys in `query`. If `rowid` identifies a previously
            incorporated member, then all data for the `rowid` will be
            automatically used as `evidence`, and therefore the only valid keys
            in `evidence` are those `output` variables for which no observation
            was incorporated.

        N : int, (optional, default None)
            Number of samples to return. If None, returns a single sample as
            a dictionary with size len(query), where each key is an `output`
            variable and each value the sample for that dimension. If `N` is
            is not None, a size N list of dictionaries will be returned, each
            corresponding to a single sample.
        """
        raise NotImplementedError

    def logpdf_score(self):
        """Return joint density of all observations and current latent state."""
        raise NotImplementedError

    def transition(self, **kwargs):
        """Apply an inference operator transitioning the internal state of CGpm.

        **kwargs : arbitrary keyword arguments Opaque binary parsed by the CGpm
            to apply inference over its latents. There are no restrictions on
            the learning mechanism, which may be based on optimization
            (variational inference, maximum likelihood, EM, etc), Markov chain
            Monte Carlo sampling (SMC, MH, etc), arbitrary heuristics, or
            others.
        """
        raise NotImplementedError

    def to_metadata(self):
        """Return the binary (json-friendly) representation of the CGpm.

        The returned B is expected to contain an entry ['factory'] which can
        be used to deserialize the binary in the following way:

        >> B = C.to_metadata()
        >> modname, attrname = B['factory']
        >> mod = importlib.import_module(modname)
        >> builder = getattr(modname, attrname)
        >> C = builder.from_metadata(binary)
        """
        raise NotImplementedError

    @staticmethod
    def from_metadata(cls, metadata, rng=None):
        """Load CGpm from its binary representation.

        Refer to the usage example in `to_metadata`.
        """
        raise NotImplementedError
