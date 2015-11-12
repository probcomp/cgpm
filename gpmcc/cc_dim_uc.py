from math import log
from math import isnan
from math import isinf
from gpmcc.cc_dim import cc_dim

class cc_dim_uc(cc_dim):
    """cc_dim. Column. Holds data, model type, and hyperparameters."""
    def __init__(self, X, cc_datatype_class, index, Z=None, n_grid=30, distargs=None):
        """
        cc_dim constructor
        input arguments:
        -- X: a numpy array of data.
        -- cc_datatype_class: the cc_type class for this column 
        -- index: the column index (int)
        optional arguments:
        -- Z: partition of data to clusters. If not specified, no clusters are 
        intialized
        -- n_grid: number of hyperparameter grid bins. Default is 30.
        -- distargs: some data types require additional information, for example
        multinomial data requires the number of multinomial categories. See the
        documentation for each type for details
        """
        super(cc_dim_uc, self).__init__(X, cc_datatype_class, index, Z=None, n_grid=30, distargs=None)
        self.mode = 'uncollapsed'
        self.params = dict()

    def singleton_predictive_logp(self,n):
        """
        Returns the predictive log_p of X[n] in its own cluster
        """
        x = self.X[n]
        lp, params = self.model.singleton_logp(x, self.hypers)
        self.params = params
        return lp


    def create_singleton_cluster(self, n, current):
        """
        Remove X[n] from clusters[current] and create a new singleton cluster
        """
        x = self.X[n]                                               # get the element
        self.clusters[current].remove_element(x)                    # remove from current cluster
        self.clusters.append(self.model(distargs=self.distargs))    # create new empty cluster
        self.clusters[-1].set_hypers(self.hypers)                   # set hypers of new cluster
        self.clusters[-1].set_params(self.params)                   # set component parameters
        self.clusters[-1].insert_element(x)                         # add element to new cluster


    def update_hypers(self):
        """
        Updates the hyperparameters and the component parameters
        """
        for cluster in self.clusters:
            cluster.set_hypers(self.hypers)
            cluster.update_component_parameters()
        self.hypers = self.clusters[0].update_hypers(self.clusters, self.hypers_grids)
        

    def reassign(self, Z):
        """
        Reassigns the data to new clusters according to the new partitioning, Z.
        Destroys and recreates dims.
        """
        self.clusters = []
        K = max(Z)+1

        for k in range(K):
            cluster = self.model(distargs=self.distargs)
            cluster.set_hypers(self.hypers)
            self.clusters.append(cluster)

        for i in range(self.N):
            k = Z[i]
            self.clusters[k].insert_element(self.X[i])

        for cluster in self.clusters:
            cluster.update_component_parameters()

        # from here down is for debugging
        assert len(self.clusters) == max(Z)+1
        N = 0
        for k in range(K):
            assert self.clusters[k].N > 0 
            N += self.clusters[k].N

        assert N == len(Z)
