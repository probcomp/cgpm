import numpy as np
from gpm import GPM
from scipy.stats import bernoulli
from cgpm.utils.datasearch_utils import compute_betabinomial_conjugate_p, flip, is_binary

class BetaBernoulli(GPM):
    def __init__(self, dim, hypers, rng):
        """
        Parameters:
        -----------
        dim : int
        hypers : list(<float>, <float>)
        rng : np.random.RandomState
        """
        self.dim = dim
        self.initial_hypers = list(hypers)
        self.suff_stats = [list(hypers) for i in range(self.dim)]
        self.rng = rng
        self.data = []
        self.check_init()

    def incorporate(self, row):
        self.check_data(row)
        self.add_row_to_data(row)
        self.update_suff_stats(row)

    def bulk_incorporate(self, data):
        for row in data:
            self.incorporate(row)

    def unincorporate(self, rowid):
        self.recover_suff_stats(rowid)
        del self.data[rowid]

    def bulk_unincorporate(self, rowid_list):
        for rowid in rowid_list:
            self.unincorporate(rowid)

    def initialize(self, data):
        if self.data:
            print "Data already initialized"
        else:
            for row in data:
                self.incorporate(row)

    def logpdf(self, query, givens=None):
        if query is None: query = {}
        if givens is None: givens = {}
        self.check_query_givens(query, givens)
        logp = 0
        for col, value in query.iteritems():
            p = compute_betabinomial_conjugate_p(self.suff_stats[col])
            logp += bernoulli._logpmf(value, p)
        assert not np.isnan(logp)
        return logp
        
    def simulate(self, target=None, givens=None, N=None):
        if target is None: target = range(self.dim)
        if givens is None: givens = {}
        self.check_target_givens(target, givens)
        if N is None: N = 1
        sample = {}
        for col in target:
            sample[col] = []
            p = compute_betabinomial_conjugate_p(self.suff_stats[col])
            for n in range(N):
                sample[col].append(flip(p, self.rng))
        return sample

    # ------------------- #
    #   NON-GPM METHODS   #
    # ------------------- #
    def add_row_to_data(self, row):
        self.check_data(row)
        self.data.append(row)

    def update_suff_stats(self, row):
        for i, value in enumerate(row):
            if not np.isnan(value):
                self.suff_stats[i][0] += value
                self.suff_stats[i][1] += 1 - value
        self.check_suff_stats()
 
    def recover_suff_stats(self, rowid):
        for i, value in enumerate(self.data[rowid]):
            if not np.isnan(value):
                self.suff_stats[i][0] -= value
                self.suff_stats[i][1] -= 1 - value
        self.check_suff_stats()

    def check_suff_stats(self):
        for phi in self.suff_stats:
            for beta in phi:
                if (np.isnan(beta) or beta < 0 or 
                    beta == np.float("inf")):
                    raise ValueError(
                        "Bad values for sufficient statistics")
                        

    def check_init(self):
        if len(self.suff_stats) != self.dim:
            raise ValueError
        if len(self.initial_hypers) != 2:
            raise ValueError

    def check_data(self, row):
        if not isinstance(row, list):
            raise TypeError
        for value in row:
            if not np.isnan(value) and not is_binary(value):
                raise ValueError
        if len(row) != self.dim:
            raise ValueError

    def check_target_givens(self, target, givens):
        if not isinstance(target, list):
            raise TypeError('target is not a list')
        if not set(target) <= set(range(self.dim)):
            raise ValueError('target is larger than %d' % (self.dim,))
        if not isinstance(givens, dict):
            raise TypeError('givens is not a dict')

    def check_query_givens(self, query, givens):
        if not isinstance(query, dict):
            raise TypeError('query is not a dict')
        for key, value in query.iteritems():
            if isinstance(value, list):
                value = value[0]
            if not is_binary(value):
                raise ValueError(
                    'queried value for col %d not binary' % (key, ))
        if not isinstance(givens, dict):
            raise TypeError('givens is not a dict')

