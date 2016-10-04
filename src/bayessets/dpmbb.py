import numpy as np
from crp import CRP
from betabernoulli import BetaBernoulli
import cgpm.utils.datasearch_utils as du
from cgpm.utils.datasearch_utils import logsumexp, max_empty

class DPMBetaBernoulli(CRP):
    def __init__(self, crp_alpha, data_dim, bb_hypers, rng=None):
        """
        Parameters:
        -----------
        crp_alpha : float, positive           
        data_dim : int
        hypers : list(float, float)
        rng : np.random.RandomState
        """
        CRP.__init__(self, crp_alpha, rng)
        self.hypers = bb_hypers
        self.dim = data_dim
        self.gpm_per_table = []
        self.data = []

    def incorporate(self, row):
        self.check_row(row)
        chosen_table = self.seat_new_customer()
        if len(self.gpm_per_table) == len(self.tables)-1:
            self.gpm_per_table.append(self.new_data_gpm())
        self.gpm_per_table[chosen_table].incorporate(row)
        self.add_row_to_data(row)

    def bulk_incorporate(self, new_data):
        for row in new_data:
            self.incorporate(row)

    def unincorporate(self, rowid):
        # del tables, gpm_per_table, customer_table if necessary
        table = self.customer_table[rowid]
        gpm = self.gpm_per_table[table]
        gpm.unincorporate(rowid)
        del self.data[rowid]
        del self.customer_table[rowid]
        if not gpm.data:
            self.check_no_customer(table)
            del self.tables[-1]
            del self.gpm_per_table[table]
            
    def initialize(self, data):
        if self.data:
            raise Warning(
                'Data already initialized. Doing nothing.')
        else:
            for row in data:
                self.incorporate(row)

    def simulate(self, target=None, givens=None, N=None):
        """
        - Sample k ~ P(z_n|givens, Z) == posterior_crp_logpdf
        - Sample target ~ P(target | z_n=k) == gpm[k].simulate(target)
        """
        if givens is None: givens = {}
        if target is None:
            target = list(set(range(self.dim)) - set(givens.keys()))
        if N is None: N = 1
        self.check_target_givens(target, givens)
        posterior_crp_logpdf = self.calc_posterior_crp_logpdf(givens)
        gpm_list = self.gpm_per_table + [self.new_data_gpm()]
        target_samples = {} 
        for n in range(N):
            k = self.logchoose_cluster(posterior_crp_logpdf)
            sampled_target = gpm_list[k].simulate(target)
            target_samples = du.merge_append_dict(
                target_samples, sampled_target)
        return target_samples
        ##TEST: simulate joint and marginal and check if they match

        ##TEST: simulate joint and conditional, select samples from
        ## joint to match conditional and check if they match

    def logpdf(self, query, givens=None):
        """
        Goal: To compute log(P(query|givens))
        - Compute log(P(z_n=k|givens, Z)) == posterior_crp_logpdf[k]
        - Compute log(P(query|z_n=k) == gpm[k].logpdf(target)
        - Compute log(P(query, z_n=k|givens, Z)) 
                  == log(P(z_n=k|givens, Z)) + log(P(target|z_n=k)
                  == joint_logpdf[k]
        - Marginalize over k : logsumexp(joint_logpdf)
        """
        if givens is None: givens = {}
        if query is None: query = {}
        self.check_query_givens(query, givens)
        posterior_crp_logpdf = self.calc_posterior_crp_logpdf(givens)
        gpm_list = self.view_gpms()
        loglikelihood = [gpm.logpdf(query) for gpm in gpm_list]
        joint_logpdf = [a+b for a, b in zip(
            posterior_crp_logpdf, loglikelihood)]
        out = du.logsumexp(joint_logpdf)
        assert not np.isnan(out)
        return out

    def logpdf_multirow(self, query, givens):
        return self.logpdf_multirow_hypothetical(query, givens)

    def logpdf_multirow_hypothetical(self, query, given_rows):
        """
        Parameters:
        -----------
        query: dict{col: value}
        given_rows: dict{rowid: dict{col: value}}
        """
        return self.logpdf_multirow_hypothetical_helper(
            i=0, query=query, given_rows=given_rows)
    
    def logpdf_multirow_hypothetical_helper(self, i, query, given_rows):
        n = len(given_rows)
        K = len(self.view_tables())
        # n_rows = len(self.data) 
        S = -np.float("inf")
        if i == n:  # base case
            S = logsumexp([S, self.logpdf(query)])  # Add givens for the final version
        else:  # recursive case
            for k in range(K):
                posterior_crp_logpdf = self.calc_posterior_crp_logpdf(
                    given_rows[i])[k] 
                self.force_incorporate(given_rows[i], cluster=k)
                S = logsumexp([
                    S, posterior_crp_logpdf +
                    self.logpdf_multirow_hypothetical_helper(
                        i+1, query, given_rows)])
                # check that all given points have been unincorporated 
                # assert len(self.customer_table) == n_rows + i + 1
                self.unincorporate(-1)
        return S
            
    # ------------------- #
    #   NON-GPM METHODS   #
    # ------------------- #
    def simulate_incorporate(self, N=None):
        if N is None: N = 1
        for n in range(N):
            row_dict = self.simulate()
            row = [row_dict[i][0] for i in range(self.dim)]
            self.incorporate(row)

    def force_incorporate(self, row, cluster):
        row = self.row_dict_to_list(row)
        self.check_row(row)
        chosen_table = self.force_seat_new_customer(cluster)
        if chosen_table == len(self.gpm_per_table):
            self.gpm_per_table.append(self.new_data_gpm())
        self.gpm_per_table[chosen_table].incorporate(row)
        self.add_row_to_data(row)

    def add_row_to_data(self, row):
        self.data.append(row)

    def calc_posterior_crp_logpdf(self, givens):
        """
        For a new point X_n, compute 
           [log (P(z_n|givens,Z)) = log (1/Z * P(givens|z_n) P(z_n|Z))
                                  ~  (givens_logweights[z_n] + crp_logpdf[z_n]) 
               for z_n = 1 ,..., k (cluster labels)]
        """
        Z = self.customer_table
        if not Z: 
            crp_logpdf = [0]
        else:
            crp_logpdf = np.log(CRP.calc_crp_weights(Z, self.alpha))
        givens_logweights = self.calc_givens_logweights(givens)
        posterior_crp_logweights = [
            a+b for a, b in zip(givens_logweights, crp_logpdf)]
        return du.lognormalize(posterior_crp_logweights)
        # TEST: implement calc_posterior_crp_pdf and compare to the above

    def calc_givens_logweights(self, givens):
        """
        Calculate [P(givens|z_n) for z_n = 1, ..., k (cluster labels)]
        """
        givens_logweights = []
        # givens_logweights for previously assigned clusters
        for i, gpm in enumerate(self.gpm_per_table):
            givens_logweights.append(gpm.logpdf(query=givens))
        # givens_logweights for new cluster
        givens_logweights.append(
            self.new_data_gpm().logpdf(
                query=givens))
        return givens_logweights
        # TEST: implement calc_givens_weights and compare to the above
 
    def logchoose_cluster(self, crp_logpdf):
        K = range(len(crp_logpdf))
        crp_pdf = np.exp(crp_logpdf)
        return self.rng.choice(K, p=crp_pdf)
        # TEST: compare to choose_cluster (with the same seed?)

    def crp_logweights(self):
        return np.log(self.table_preferences())

    def new_data_gpm(self):
        return BetaBernoulli(self.dim, self.hypers, self.rng)

    def view_gpms(self):
        "Retrieve the currently availabe gpms and add one empty gpm"
        return self.gpm_per_table + [self.new_data_gpm()]

    def row_dict_to_list(self, row):
        if isinstance(row, dict):
            M = max(row.keys())
            if M > self.dim:
                raise ValueError("too many columns in row")
            out = [np.nan] * self.dim
            for i in row.keys():
                out[i] = row[i]
            row = out
        return row

    # -------------------- #
    #   CHECKING METHODS   #
    # -------------------- #
    def check_row(self, row):
        if not isinstance(row, (list, dict)):
            raise TypeError
        for value in row:
            if not np.isnan(value) and not du.is_binary(value):
                raise ValueError
        if len(row) != self.dim:
            raise ValueError("Bad number of columns in row")

    def check_cluster(self, cluster):
        if not isinstance(cluster, int):
            raise TypeError
        if cluster > len(self.tables) + 1:
            raise ValueError, 'cluster not contiguous to current clusters'

    def check_no_customer(self, table):
        if table in self.customer_table:
            raise Exception('Table is not empty.')

    def check_target_givens(self, target, givens):
        if not isinstance(target, list):
            raise TypeError('target is not a list')
        if not set(target) <= set(range(self.dim)):
            raise ValueError('target is larger than %d' % (self.dim,))
        if not isinstance(givens, dict):
            raise TypeError('givens is not a dict')
        givens_keys = givens.keys()
        if not set(target).isdisjoint(givens_keys):
            raise ValueError('Target and givens must be disjoint')

    def check_query_givens(self, query, givens):
        if not isinstance(query, dict):
            raise TypeError('query is not a dict')
        for key, value in query.iteritems():
            if isinstance(value, list):
                value = value[0]
            if not du.is_binary(value):
                raise ValueError(
                    'queried value for col %d not binary' % (key, ))
        if not isinstance(givens, dict):
            raise TypeError('givens is not a dict')
        query_keys = query.keys()
        givens_keys = givens.keys()
        if not set(query_keys).isdisjoint(givens_keys):
            raise ValueError('Query and givens must be disjoint')
