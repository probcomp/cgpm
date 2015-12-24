# -*- coding: utf-8 -*-

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

class DistributionGpm(object):
    """Interface for generative population models representing a
    probability distribution over a single dimension.

    A typical ComponentGpm will have:
    - Sufficient statistics T, for the observed data X.
    - Parameters Q, for the likelihood p(X|Q).
    - Hyperparameters H, for the prior p(Q|H).

    This interface is uniform for both collapsed and uncollapsed models.
    A collapsed model will typically have no parameters Q.
    An uncollapsed model will typically carry a single set of parameters Q,
    but may also carry an ensemble of parameters (Q1,...,Qn) for simple
    Monte Carlo averaging of queries. The collapsed case is "theoretically"
    recovered in the limit n \to \infty.
    """

    def __init__(self, suffstats, params, hypers):
        """Initialize the Gpm.

        This constructor signature is abstract. `suffstats`, `params`, and
        `hypers` will be unrolled into actual names by each GPM, but the
        order of arguments must be as above. The idea is that
        a user who has suffstats, params, and hypers in named dicts
        can successfully invoke the consructor with
            __init__(N, **suffstats, **params, **hypers) should work.

        Keyword Arguments:
        ... suffstats : Initial values of suffstats.
        ... params : Initial values of params.
        ... hypers : Initial values of hyperparams.
        """
        raise NotImplementedError

    def incorporate(self, x):
        """Record a single observation x, increment any suffstats."""
        raise NotImplementedError

    def unincorporate(self, x):
        """Remove a single observation x, decrement any suffstats.
        An error will be thrown if `self.N` drops below zero.
        """
        raise NotImplementedError

    def predictive_logp(self, x):
        """Compute the probability of a new observation x, conditioned on
        the sufficient statistics, parameters (if uncollapsed), and
        hyperparameters. P(x|T,Q,H).
        """
        raise NotImplementedError

    def singleton_logp(self, x):
        """Compute the probability of a new observation x, conditioned on
        parameters (if uncollapsed), and hyperparameters. P(x|Q,H). Note
        that previous observations (suffstats) are ignored in this
        computation.
        """
        raise NotImplementedError

    def marginal_logp(self):
        """Compute an estimate of the probability of all incorporated
        observations X, conditioned on the current GPM state.

        A collapsed model can compute P(X|H) exactly by integrating over Q.
        An uncollapsed model might approximate this marginal by
            P(X|H) \appx avg{P(X|Qi)P(Qi|H) for i = 1 .. N}

        While this probability is necessarily a function of the data only
        through the sufficient statistics, the probability is
        **not the probability of the sufficient statistics**
        rather the exchangeable sequence of observations that are
        summarized by the sufficient statistics.
        """
        raise NotImplementedError

    def simulate(self):
        """Simulate from the posterior predictive_logp p(x|T,Q,H). The
        sample returned by this method must necessarily be from the same
        distribution of `predictive_logp`.
        """
        raise NotImplementedError

    def transition_params(self, x):
        """Resample the parameters Q conditioned on all observations X
        from an approximate posterior P(Q|X,H)."""
        raise NotImplementedError

    def set_hypers(self, hypers):
        """Force the hyperparameters H to new values."""
        raise NotImplementedError

    def get_hypers(self):
        """Return a dictionary of hyperparameters."""
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
    def plot_dist(X, clusters, ax=None, Y=None, hist=True):
        """Plot the mixture distribution of the DistributionGpm
        represented by `clusters`. The weight of each cluster is
        proportional to its number of observations N.

        Parameters
        ----------
        X : list
            List of samples from the empirical distribution to plot, very
            poorly named for historical reasons.
        clusters : list
            List of DistributionGpm objects, all of the same type.
        ax : matplotlib.axes, optional
            Axes object on which to plot the distribution. If None, a new
            axis will be created.
        Y : list, optional
            Values on which to evaluate the probability density function.
        hist : bool, optional
            Show a histogram of samples X? Otherwise samples will be shown
            as small vertical lines.
        """
        raise NotImplementedError

    @staticmethod
    def name():
        """Return the name of the distribution as a string."""
        raise NotImplementedError

    @staticmethod
    def is_collapsed():
        """Is the sampler collapsed?."""
        raise NotImplementedError
