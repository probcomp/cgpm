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

from gpmcc.dim import Dim

class DimUC(Dim):
    """Holds data, model type, and hyperparameters."""

    def __init__(self, X, cc_datatype_class, index, Z=None, n_grid=30,
            distargs=None):
        """DimUC constructor.

        Input arguments:
        -- X: a numpy array of data.
        -- cc_datatype_class: the cc_type class for this column.
        -- index: the column index (int).

        Optional arguments:
        -- Z: partition of data to clusters. If not specified, no clusters are
        intialized.
        -- n_grid: number of hyperparameter grid bins. Default is 30.
        -- distargs: some data types require additional information, for example
        multinomial data requires the number of multinomial categories. See the
        documentation for each type for details.
        """
        super(DimUC, self).__init__(X, cc_datatype_class, index, Z=Z,
            n_grid=n_grid, distargs=distargs)
        self.mode = 'uncollapsed'

    def resample_hypers(self):
        """Updates the hyperparameters and the component parameters."""
        for cluster in self.clusters:
            cluster.resample_params()
        self.hypers = self.model.resample_hypers(self.clusters,
            self.hypers_grids)
        for cluster in self.clusters:
            cluster.set_hypers(self.hypers)
