# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

class DistributionGpm(object):
    """Interface for generative population models representing a
    probability distribution over a single dimension, possibly conditioned
    on other dimensions.

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

    def __init__(self, suffstats, params, hypers, distargs):
        """Initialize the Gpm.

        This constructor signature is abstract. `suffstats`, `params`, and
        `hypers` will be unrolled into actual names by each GPM, but the
        order of arguments must be as above. The idea is that
        a user who has suffstats, params, and hypers in named dicts
        can successfully invoke the consructor with
            __init__(N, **suffstats, **params, **hypers) should work.

        Parameters
        ----------
        suffstats : **kwargs
            Initial values of suffstats.
        params : **kwargs
            Initial values of params.
        hypers : **kwargs
            Initial values of hyperparams.
        distargs : dict
            Any other arguments.
        """
        raise NotImplementedError

    def incorporate(self, x, y=None):
        """Record a single observation x|y. Increments suffstats."""
        raise NotImplementedError

    def unincorporate(self, x, y=None):
        """Remove a single observation x|y. Decrements any suffstats.
        An error will be thrown if `self.N` drops below zero or other
        distribution-specific invariants are violated.
        """
        raise NotImplementedError

    def logpdf(self, x, y=None):
        """Compute the probability of a new observation x|y, conditioned on
        the sufficient statistics, parameters (if uncollapsed), and
        hyperparameters, ie P(x|T,Q,H).
        """
        raise NotImplementedError

    def logpdf_marginal(self):
        """Compute an estimate of the probability of all incorporated
        observations X|Y, conditioned on the current GPM state.

        A collapsed model can compute P(X|H) exactly by integrating over Q.

        An uncollapsed model might approximate this marginal by
            Qi ~ P(Qi|H) i = 1 .. N
            P(X|H) \appx avg{P(X|Qi)P(Qi|H) for i = 1 .. N}

        While this probability is necessarily a function of the data only
        through the sufficient statistics, the probability is **not the
        probability of the sufficient statistics**. It is the probablility of
        an exchangeable sequence of observations that are summarized by the
        sufficient statistics. Implemented as an optimization, and can be
        recovered by unincorporating all data, and can be recovered by invoking
        `logpdf` after each incorporate and taking the sum.
        """
        raise NotImplementedError

    def simulate(self, y=None):
        """Simulate from the distribution p(x|T,Q,H,Y=y). The sample returned
        by this method must necessarily be from the same distribution that
        `predictive_logp` evaluates.

        A collapsed sampler will typically simulate by marginalizing over
        parameters Q.

        An uncollapsed sampler will typically require a call to
        `transition_params` for observations to have an effect on the
        simulation of the next sample.
        """
        raise NotImplementedError

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
        """Is the support of the pdf a numeric of symbolic set?"""
        raise NotImplementedError
