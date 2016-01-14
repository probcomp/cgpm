# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import multiprocessing

import gpmcc.utils.plots as pu
import gpmcc.utils.general as gu
import gpmcc.utils.validation as vu

from gpmcc.state import State

_transition_kernels = ['column_z','state_alpha', 'row_z', 'column_hypers',
    'view_alphas']

def _transition((N, kernel_list, target_rows, target_cols, metadata)):
    chain = State.from_metadata(metadata)
    chain.transition(N, kernel_list, target_rows, target_cols)
    return chain.get_metadata()

def _intialize((X, cctypes, distargs, seed)):
    chain = State(X, cctypes, distargs, seed=seed)
    return chain.get_metadata()

class Engine(object):
    """Multiprocessing engine for StateGPMs running in parallel."""

    def __init__(self):
        self.metadata = None

    def initialize(self, X, cctypes, distargs, num_states=1,
            col_names=None, seeds=None, multithread=True):
        """Initialize `num_states` States."""
        vu.validate_data(X)
        vu.validate_cctypes(cctypes)

        self.X = X
        self.cctypes = cctypes
        self.num_states = num_states
        self.n_rows = len(X[0])
        self.n_cols = len(X)

        if col_names is None:
            col_names = [ "col%i" % i for i in xrange(len(X))]
        self.col_names = col_names

        if seeds is None:
            seeds = range(num_states)
        self.seeds = seeds

        _, mapper = self._get_mapper(multithread=multithread)
        args = ((X, cctypes, distargs, seed) for seed in seeds)
        self.metadata = mapper(_intialize, args)

    def get_state(self, index):
        """Return an individual state from the ensemble."""
        return State.from_metadata(self.metadata[index])

    def transition(self, N=1, kernel_list=None, target_rows=None,
            target_cols=None, multithread=True):
        """Run transitions in parallel."""
        _, mapper = self._get_mapper(multithread=multithread)
        args = [(N, kernel_list, target_rows, target_cols,
            self.metadata[i]) for i in xrange(self.num_states)]
        self.metadata = mapper(_transition, args)

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

    def incorporate_dim(self, X_f, cctype, distargs=None, m=1,
            col_name=None):
        """Add a feature (column) to the data."""
        raise NotImplementedError()

    def incorporate_row(self, X_o, update_hypers_grid=False):
        """Add a (row) to the cc_state."""
        raise NotImplementedError()

    def _get_mapper(self, multithread=True):
        pool, mapper = None, map
        if multithread:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            mapper = pool.map
        return pool, mapper
