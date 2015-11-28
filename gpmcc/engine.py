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

import gpmcc.utils.plots as pu
import gpmcc.utils.general as gu
import gpmcc.utils.validation as vu

from gpmcc.state import State

import multiprocessing

_transition_kernels = ['column_z','state_alpha', 'row_z', 'column_hypers',
    'view_alphas']

def _append_feature(args):
    X_f, cctype, distargs, metadata = args
    chain = State.from_metadata(metadata)
    chain.append_dim(X_f, cctype, distargs=distargs, m=1)
    return chain.get_metadata()

def _transition(args):
    N, kernel_list, target_rows, target_cols, metadata = args
    chain = State.from_metadata(metadata)
    chain.transition(N, kernel_list, target_rows, target_cols)
    return chain.get_metadata()

def _intialize(args):
    X, cctypes, distargs, seed = args
    chain = State(X, cctypes, distargs, seed=seed)
    return chain.get_metadata()

class Engine(object):
    """Engine."""

    def __init__(self, multithread=True):
        self.multithread = multithread
        self.map = map
        if self.multithread:
            self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.map = self.pool.map
        self.metadata = None

    def initialize(self, X, cctypes, distargs, num_states=1, col_names=None,
            seeds=None):
        """Get or set the initial state."""
        vu.validate_data(X)
        vu.validate_cctypes(cctypes)
        self.X = X
        self.cctypes = cctypes
        self.num_states = num_states
        self.n_rows = len(X[0])
        self.n_cols = len(X)

        if col_names is None:
            col_names = [ "col_" + str(i) for i in xrange(len(X))]
        else:
            assert isinstance(col_names, list)
            assert len(col_names) == len(X)
        self.col_names = col_names

        if seeds is None:
            seeds = range(num_states)
        else:
            assert len(seeds) == num_states
            assert float(seeds) == int(seeds)
        self.seeds = seeds

        args = ((X, cctypes, distargs, seed) for (_, seed) in
            zip(xrange(self.num_states), seeds))
        self.metadata = self.pool.map(_intialize, args)

    def initialize_csv(self, cctypes, distargs, filename, num_states=1,
            seed=0):
        X, col_names = gu.csv_to_data_and_colnames(filename)
        self.initialize(X.T, cctypes, distargs, num_states=num_states,
            col_names=col_names, seed=seed)

    def get_state(self, index):
        return State.from_metadata(self.metadata[index])

    def append_feature(self, X_f, cctype, distargs=None, m=1,
            col_name=None):
        """Add a feature (column) to the data."""
        # Check data size.
        if X_f.shape[0] != self.n_rows:
            raise ValueError("X_f has %i rows; should have %i." % \
                (X_f.shape[0], self.n_rows))

        # Check that cc_type is valid.
        vu.validate_data(X_f)
        vu.validate_cctypes(cctype)

        args = [(self.X, X_f, cctype, distargs, m, self.metadata[i])
            for i in range(self.num_states) ]

        self.metadata = self.map(self._append_feature, args)

        self.X.append(X_f)
        self.cctypes.append(cctype)

        if col_name is None:
            self.col_names.append( 'col_' + str(len(self.X)-1) )
        else:
            self.col_names.append(col_name)

    def append_row(self, X_o, update_hypers_grid=False):
        """Add an object (row) to the cc_state."""
        pass

    def transition(self, N=1, kernel_list=None, target_rows=None,
            target_cols=None):
        """Do transitions in parallel."""
        args = [(N, kernel_list, target_rows, target_cols,
            self.metadata[i]) for i in xrange(self.num_states)]
        self.metadata = self.map(_transition, args)

    def predictive_probabiilty(self, query, constraints=None):
        """Predictive probability."""

    def predictive_sample(self, query, constraints=None):
        """Predictive sample."""

    def impute(self, query):
        """Impute data."""

    def plot_Z(self):
        """Plot data in different ways."""
        Zvs = [metadata['Zv'] for metadata in self.metadata]
        col_names = list(self.col_names)
        pu.generate_Z_matrix(Zvs, col_names)

    @staticmethod
    def _predictive_sample():
        pass

    @staticmethod
    def _predictive_probability():
        pass
