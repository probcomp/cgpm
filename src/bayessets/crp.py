import numpy as np
import matplotlib.pyplot as plt

class CRP(object):
    def __init__(self, alpha, rng=None):
        self.alpha = alpha
        if rng is None: rng = np.random.RandomState()
        self.rng = rng
        self.customer_table = []
        self.tables = []
        
    def seat_new_customer(self):  # seat_new_customer -> seat_new_customer
        if not self.tables:
            chosen_table = 0
        else: 
            chosen_table = self.choose_table()
        self.customer_table.append(chosen_table)
        if chosen_table > len(self.tables)-1:
            self.tables.append(chosen_table)
        return chosen_table

    def force_seat_new_customer(self, table):
        if table > len(self.tables):
            raise ValueError("Cannot skip table")
        chosen_table = table
        self.customer_table.append(chosen_table)
        if chosen_table > len(self.tables)-1:
            self.tables.append(chosen_table)
        return chosen_table

    def choose_table(self):
        available_tables = self.view_tables()
        preference = self.table_preferences()
        chosen_table = self.rng.choice(available_tables, p=preference)
        return chosen_table

    def view_tables(self):
        "Each customer sees all occupied tables and a single unoccupied one."
        return range(len(self.tables) + 1)

    def table_preferences(self):
        return CRP.calc_crp_weights(self.customer_table, self.alpha)
    
    def self_check(self):
        assert len(set(np.diff(self.tables))) == 1
        assert max(self.customer_table) == self.tables[-1]
        assert min(self.customer_table) == 0
    
    def histogram(self):
        self.self_check()
        fig, ax = plt.subplots()
        ax.hist(self.customer_table, bins=self.tables[-1]+1)
        
    def plot_horizontal(self, ax=None, skip_sticks=0):
        self.self_check()
        if ax is None: 
            _, ax = plt.subplots()
        ax.hlines(0, 0, 1)
        rho = self.table_preferences() 
        first_stick = sum(rho[:skip_sticks])
        splinter_start = first_stick
        for i, rho in enumerate(rho[skip_sticks:-1]):
            splinter_end = splinter_start + rho
            ax.hlines(-i, splinter_start, splinter_end, linewidths=5)
            splinter_start= splinter_end
        ax.set_ylim(-i -1, 1)
        ax.set_xlim(first_stick,1)
        if skip_sticks:
            ax.set_title("StickBreaking: Skipped %d tables" % (skip_sticks,))

    @staticmethod
    def calc_crp_weights(Z, alpha):
        """
        Parameters:
        -----------
        Z : list(<int>),
            Z[i] is the cluster lable for i-th data point.

        P(Z[N+1]|Z) == alpha / (N+alpha),      if Z[N+1]==K   (new cluster)
                    == Z.count(i) / (N+alpha), if Z[N+1]==i<K (old cluster)
        """
        if not Z:
            raise ValueError("Z cannot be empty")
        if set(Z) != set(range(max(Z)+1)):
            raise ValueError("Z implies empty clusters")
        K = max(Z)+1
        N = len(Z)
        weights = []
        for i in range(K+1):
            if i == K:
                weights.append(alpha*1. / (N+alpha))
            else:
                n_i = Z.count(i)
                weights.append(n_i*1. / (N+alpha))
        if not np.allclose(sum(weights), 1.):
            raise ValueError(
                "Weights do not sum to one:\n%s" % (weights,))
        return weights
            

class DPMixtureCRP(CRP):
    def __init__(self, sample_theta, sample_likelihood, alpha, rng):
        CRP.__init__(self, alpha, rng)
        self.sample_likelihood = sample_likelihood
        self.sample_theta = sample_theta
        self.theta = []
        
    def simulate_point(self):
        theta, i = self.simulate_theta()
        return self.sample_likelihood(theta), i  
    
    def seat_new_customer(self):
        CRP.seat_new_customer(self)
        if len(self.theta) == len(self.tables)-1:
            self.theta.append(self.sample_theta())

    def simulate_theta(self):
        self.seat_new_customer()
        i = self.customer_table[-1]
        return self.theta[i], i
