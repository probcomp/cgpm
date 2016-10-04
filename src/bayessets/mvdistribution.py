# Modified from: https://github.com/probcomp/cgpm/src/exponentials/distribution.py
from cgpm.utils import general as gu
from cgpm.cgpm import CGpm


class MultivariateDistributionGpm(CGpm):
    """Interface for generative population models representing multivariate
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
        assert not inputs
        if outputs is None:
            outputs = len(params.get('p', [1]))
        self.outputs = range(outputs)
        self.D = len(self.outputs)
        self.inputs = []
        self.data = dict()
        self.rng = gu.gen_rng() if rng is None else rng

    def incorporate(self, rowid, query, evidence=None):
        assert rowid not in self.data
        assert not evidence
        assert query.keys() in self.outputs
 
    def logpdf(self, rowid, query, evidence=None):
        assert rowid not in self.data
        assert not evidence
        if isinstance(query, dict):
            assert query.keys() in self.outputs
        elif isinstance(query, list):
            assert len(query) == self.D
        else:
            raise TypeError("query must be of type <list> or <dict>")

    def simulate(self, rowid, query, evidence=None, N=None):
        assert not evidence
        if query is None: query = self.outputs
        assert query in self.outputs

    ##################
    # NON-GPM METHOD #
    ##################

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
        """Is the support of the pdf a numeric of symbolic set?"""
        raise NotImplementedError
