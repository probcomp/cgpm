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

def load_docstrings(module):
    module.State.__init__.__func__.__doc__ = """
        Construct a State.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, each row is an observation and each column a variable.
        outputs : list<int>, optional
            Unique non-negative ID for each column in X, and used to refer to
            the column for all future queries. Defaults to range(1, X.shape[1])
        inputs : list<int>, optional
            Currently unsupported.
        cctypes : list<str>, optional
            Data type of each column, see `utils.config` for valid cctypes.
            Defaults to normal.
        distargs : list<dict>, optional
            See the documentation for each DistributionGpm for its distargs.
        Zv : dict{int:int}, optional
            Assignmet of columns to views. Defaults to sampling from CRP.
        Zrv : dict{int:list<int>}, optional
            Assignment of rows to clusters in each view, where Zrv[k] is
            the Zr for view k. If specified, then Zv must also be specified.
            Defaults to sampling from CRP.
        Cd : list(list<int>), disabled
            List of marginal dependence constraints for columns. Each element in
            the list is a list of columns which are to be in the same view. Each
            column can only be in one such list i.e. [[1,2,5],[1,5]] is not
            allowed.
        Ci : list(tuple<int>), optional
            List of marginal independence constraints for columns.
            Each element in the list is a 2-tuple of columns that must be
            independent, i.e. [(1,2),(1,3)].
        Rd : dict(int:Cd), disabled
            Dictionary of dependence constraints for rows, wrt.
            Each entry is (col: Cd), where col is a column number and Cd is a
            list of dependence constraints for the rows with respect to that
            column (see doc for Cd).
        Ri : dict(int:Cid), disabled
            Dictionary of independence constraints for rows, wrt.
            Each entry is (col: Ci), where col is a column number and Ci is a
            list of independence constraints for the rows with respect to that
            column (see doc for Ci).
        iterations : dict(str:int), optional
            Metadata holding the number of iters each kernel has been run.
        rng : np.random.RandomState, optional.
            Source of entropy.
        """


    # --------------------------------------------------------------------------
    # Observe

    module.State.incorporate_dim.__func__.__doc__ = """
        Incorporate a new Dim into this State.

        Parameters
        ----------
        T : list
            Data with length self.n_rows().
        outputs : list[int]
            Identity of the variable modeled by this dim, must be non-negative
            and cannot collide with State.outputs. Only univariate outputs
            currently supported.
        inputs : list[int], disabled
            List of input variables.
        cctype: str, optional
            Name of the CGPM to model this dimension.
        distargs: dict, optional
            Additional configuration to `cctype`.
        v : int, optional
            Index of the view to assign the data. If 0 <= v < len(state.views)
            then insert into an existing View. If v = len(state.views) then
            singleton view will be created with a partition from the CRP prior.
            If unspecified, will be sampled.
        """

    module.State.unincorporate_dim.__func__.__doc__ = """
        Remoe an existing Dim from this State.

        Parameters
        ----------
        col : int
            Index of the dimension to remove. An error will be thrown if `col`
            is the only variable.
        """

    module.State.incorporate.__func__.__doc__ = """
        Incorporate a new member with token rowid.

        Parameters
        ----------
        rowid : int
            Only rowid = -1 is currently supported, pending Github #84 which
            will support cell-level operations.
        query : dict{output:val}
            Keys of the query must a subset of the State output, unspecified
            outputs will be nan. At least one non-nan value must be specified.
            Optionally use {-v:k} for latent cluster assignments of rowid, where
            1 <= v <= len(State.views) and 0 <= k <= len(State.views[v].Nk).
        """


    # --------------------------------------------------------------------------
    # Schema updates.

    module.State.update_cctype.__func__.__doc__ = """
        Update the distribution type of self.dims[col] to cctype.

        Parameters
        ----------
        col : int
            Index of column to update.
        """


    # --------------------------------------------------------------------------
    # logpdf

    module.State.logpdf.__func__.__doc__ = """
        Compute density of query under the posterior predictive distirbution.

        Parameters
        ----------
        rowid : int
            The rowid of the member of the population to simulate from.
            If 0 <= rowid < state.n_rows then the latent variables of member
            rowid will be taken as conditioning variables.
            Otherwise logpdf for a hypothetical member is computed,
            marginalizing over latent variables.
        query : list(tuple<int>)
            List of pairs (col, val) at which to query the logpdf.
        evidence : list(tuple<int>), optional
            List of pairs (col, val) of conditioning values in the row.

        Returns
        -------
        logpdf : float
            The logpdf(query|rowid, evidence).
        """

    # --------------------------------------------------------------------------
    # Simulate

    module.State.simulate.__func__.__doc__ = """
        Simulate from the posterior predictive distirbution.

        Parameters
        ----------
        rowid : int
            The rowid of the member of the population to simulate from.
            If 0 <= rowid < state.n_rows then the latent variables of member
            rowid will be taken as conditioning variables.
            Otherwise a hypothetical member is simulated, marginalizing over
            latent variables.
        query : list<int>
            A list of col numbers to simulate from.
        evidence : list(tuple<int>), optional
            A list of pairs (col, val) of observed values in the row to
            condition on.
        N : int, optional.
            Number of samples to return.

        Returns
        -------
        samples : np.array
            A N x len(query) array, where samples[i] ~ P(query|rowid, evidence).
        """

    # --------------------------------------------------------------------------
    # Mutual information

    module.State.mutual_information.__func__.__doc__ = """
        Computes the mutual information MI(col0:col1|evidence).

        Mutual information with conditioning variables can be interpreted in
        two forms
            - MI(X:Y|Z=z): point-wise CMI, (this function).
            - MI(X:Y|Z): expected pointwise CMI E_Z[MI(X:Y|Z)] under Z.

        The rowid is hypothetical. For any observed member, the rowid is
        sufficient and decouples all columns.

        Parameters
        ----------
        col0, col1 : int
            Columns to comptue MI. If col0 = col1 then entropy is returned.
        evidence : list(tuple<int>), optional
            A list of pairs (col, val) of observed values to condition on.
        N : int, optional.
            Number of samples to use in the Monte Carlo estimate.

        Returns
        -------
        mi : float
            A point estimate of the mutual information.
        """

    module.State.conditional_mutual_information.__func__.__doc__ = """
        Computes conditional mutual information MI(col0:col1|evidence).

        Mutual information with conditioning variables can be interpreted in
        two forms
            - MI(X:Y|Z=z): point-wise CMI.
            - MI(X:Y|Z): expected pointwise CMI E_z[MI(X:Y|Z=z)] under Z
            (this function).

        The rowid is hypothetical. For any observed member, the rowid is
        sufficient and decouples all columns.

        Parameters
        ----------
        col0, col1 : int
            Columns to comptue MI. If col0 = col1 then entropy is returned.
        evidence : list<int>
            A list of columns to condition on.
        T : int, optional.
            Number of samples to use in external Monte Carlo estimate (z~Z).
        N : int, optional.
            Number of samples to use in internal Monte Carlo estimate.

        Returns
        -------
        mi : float
            A point estimate of the mutual information.
        """

    # --------------------------------------------------------------------------
    # Inference

    module.State.transition.__func__.__doc__ = """
        Run targeted inference kernels.

        Parameters
        ----------
        N : int, optional
            Number of iterations to transition. Default 1.
        S : float, optional
            Number of seconds to transition. If both N and S set then min used.
        kernels : list<{'alpha', 'view_alphas', 'column_params', 'column_hypers'
            'rows', 'columns'}>, optional
            List of inference kernels to run in this transition. Default all.
        views, rows, cols : list<int>, optional
            View, row and column numbers to apply the kernels. Default all.
        do_plot : boolean, optional
            Plot the state of the sampler (real-time), 24 columns max. Unstable.
        progress : boolean, optional
            Show a progress bar for number of target iterations or elapsed time.
        """
