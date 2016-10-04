class GPM(object):
    """ Interface for generative population models"""

    
    def __init__(self, schema, rng):
        """Initialize the Gpm.

        Parameters
        ----------
        schema : 
        rng : numpy.random.RandomState
            Source of entropy.
        """
        raise NotImplementedError

    def incorporate(self, row):
        """Record an observation for row.
        Adds the row in to the model, sampling the category assignment
        from its conditional distribution given any existing
        categories in the model, and (for each observed column)
        updates the sufficient statistics for the Beta-Beronulli
        component models for the chosen category

        rowid : int
            A unique integer identifying the member.
        query : dict{int:value}
            The observed values. The keys of `query` must be a subset of the
            `output` variables, and `value` must be type-matched based on
            `schema`.
        """
        raise NotImplementedError

    def unincorporate(self, rowid):
        """Remove all incorporated observations of `rowid`."""
        raise NotImplementedError

    def initialize(self, categorization, dataset):
        """call "incorporate" on each row of the dataset"""
        raise NotImplementedError

    def logpdf(self, query, givens=None):
        """Return the conditional density of `query` given `evidence`.
        
         using enumeration to explicitly create and sum over all the
         category assignments for the rows

        query : dict{int:value}
        givens : dict{int:value}
        """
        raise NotImplementedError

    def simulate(self, target, givens=None, N=None):
        """Return N iid samples of `target` variables conditioned on `givens`.

         using enumeration to explicitly create and sum over all the
         category assignments for the rows

        The sample must be drawn from the same distribution whose density is
        assessed by `logpdf`.

        target : list<int>
            List of variables to simulate.
        givens : dict{int:value}, optional
        N : int, optional
            Number of samples to return. If None, returns a single sample. If
            integer, results a list of samples of length N.
        """
        raise NotImplementedError

