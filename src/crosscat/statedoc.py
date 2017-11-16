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
            the column for all future queries. Defaults to range(0, X.shape[1])
        inputs : list<int>, optional
            Currently unsupported.
        cctypes : list<str>
            Data type of each column, see `utils.config` for valid cctypes.
        distargs : list<dict>, optional
            See the documentation for each DistributionGpm for its distargs.
        Zv : dict(int:int), optional
            Assignment of output columns to views, where Zv[k] is the
            view assignment for column k. Defaults to sampling from CRP.
        Zrv : dict(int:list<int>), optional
            Assignment of rows to clusters in each view, where Zrv[k] is
            the Zr for View k. If specified, then Zv must also be specified.
            Defaults to sampling from CRP.
        Cd : list(list<int>), optional
            List of marginal dependence constraints for columns. Each element in
            the list is a list of columns which are to be in the same view. Each
            column can only be in one such list i.e. [[1,2,5],[1,5]] is not
            allowed.
        Ci : list(tuple<int>), optional
            List of marginal independence constraints for columns.
            Each element in the list is a 2-tuple of columns that must be
            independent, i.e. [(1,2),(1,3)].
        Rd : dict(int:Cd), optional
            Dictionary of dependence constraints for rows, wrt.
            Each entry is (col: Cd), where col is a column number and Cd is a
            list of dependence constraints for the rows with respect to that
            column (see doc for Cd).
        Ri : dict(int:Cid), optional
            Dictionary of independence constraints for rows, wrt.
            Each entry is (col: Ci), where col is a column number and Ci is a
            list of independence constraints for the rows with respect to that
            column (see doc for Ci).
        iterations : dict(str:int), optional
            Metadata holding the number of iters each kernel has been run.
        loom_path: str, optional
            Path to a loom project compatible with this State.
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
            currently supported, so the list be a singleton.
        cctype, distargs:
            refer to State.__init__
        v : int, optional
            Index of the view to assign the data. If 0 <= v < len(state.views)
            then insert into an existing View. If v = len(state.views) then
            singleton view will be created with a partition from the CRP prior.
            If unspecified, will be sampled.
        """

    module.State.incorporate.__func__.__doc__ = """
        Incorporate a new rowid.

        Parameters
        ----------
        rowid : int
            Only rowid = -1 is currently supported, pending Github #84 which
            will support cell-level operations.
        observation : dict{output:val}
            Keys of the query must a subset of the State output, unspecified
            outputs will be nan. At least one non-nan value must be specified.
            To optionally specify the cluster assignment in a particular view v,
            include a key in the query with state.views[v].outputs[0] whose
            value is the desired cluster id.
        """


    # --------------------------------------------------------------------------
    # Schema updates.

    module.State.update_cctype.__func__.__doc__ = """
        Update the distribution type of self.dims[col] to cctype.

        Parameters
        ----------
        col : int
            Index of column to update.
        cctype, distargs:
            refer to State.__init__
        """


    # --------------------------------------------------------------------------
    # Compositions

    module.State.compose_cgpm.__func__.__doc__ = """
        Compose a CGPM with this object.

        Parameters
        ----------
        cgpm : cgpm.cgpm.CGpm object
            The `CGpm` object to compose.

        Returns
        -------
        token : int
            A unique token representing the composed cgpm, to be used
            by `State.decompose_cgpm`.
    """

    module.State.decompose_cgpm.__func__.__doc__ = """
        Decompose a previously composed CGPM.

        Parameters
        ----------
        token : int
            The unique token representing the composed cgpm, returned from
            `State.compose_cgpm`.
    """


    # --------------------------------------------------------------------------
    # logpdf_score

    module.State.logpdf_score.__func__.__doc__ = """
        Compute joint density of all latents and the incorporated data.

        Returns
        -------
        logpdf_score : float
            The log score is P(X,Z) = P(X|Z)P(Z) where X is the observed data
            and Z is the entirety of the latent state in the CGPM.
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
        query : dict{col:val}
            Columns and values at which to query the logpdf.
        evidence : dict{col:val}
            Columns and values at which serve as conditioning values in the row.

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
            A list of columns to simulate from.
        evidence : dict{col:val}
            Columns and values at which serve as conditioning values in the row.
        N : int, optional.
            Number of samples to return.

        Returns
        -------
        samples : dict or list<dict>
            If `N` is `None`, returns a scalar dictionary whose keys are the
            queried columns, and values are the simulations. If `N` is an
            integer, returns a list where each item is a dictionary representing
            a single joint sample.
        """

    # --------------------------------------------------------------------------
    # Mutual information

    module.State.mutual_information.__func__.__doc__ = """
        Computes the mutual information MI(col0:col1|evidence).

        Mutual information with evidence can be of the form:
            - MI(X:Y|Z=z): CMI at a fixed conditioning value.
            - MI(X:Y|Z): expected CMI E_Z[MI(X:Y|Z)] under Z.
            - MI(X:Y|Z, W=w): expected CMI E_Z[MI(X:Y|Z,W=w)] under Z.

        This function supports all three forms. The CMI is computed under the
        posterior predictive joint distributions.

        Parameters
        ----------
        col0, col1 : list<int>
            Columns to comptue MI. If all columns in `col0` are equivalent
            to columns in `col` then entropy is returned, otherwise they must
            be disjoint and the CMI is returned
        evidence : list(tuple), optional
            A list of pairs (col, val) of observed values to condition on. If
            `val` is None, then `col` is marginalized over.
        T : int, optional.
            Number of samples to use in the outer (marginalization) estimator.
        N : int, optional.
            Number of samples to use in the inner Monte Carlo estimator.

        Returns
        -------
        mi : float
            A point estimate of the mutual information.

        Examples
        -------
        # Compute MI(X:Y)
        >>> State.mutual_information(col_x, col_y)
        # Compute MI(X:Y|Z=1)
        >>> State.mutual_information(col_x, col_y, {col_z: 1})
        # Compute MI(X:Y|W)
        >>> State.mutual_information(col_x, col_y, {col_w:None})
        # Compute MI(X:Y|Z=1, W)
        >>> State.mutual_information(col_x, col_y, {col_z: 1, col_w:None})
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
        checkpoint : int, optional
            Number of transitions between recording inference diagnostics
            from the latent state (such as logscore and row/column partitions).
            Defaults to no checkpointing.
        progress : boolean, optional
            Show a progress bar for number of target iterations or elapsed time.
        """
