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

import itertools
import multiprocessing
import pickle

import numpy as np

from gpmcc.state import State

_transition_kernels = ['column_z','state_alpha', 'row_z', 'column_hypers',
    'view_alphas']

def _transition((N, kernel_list, target_rows, target_cols, metadata)):
    chain = State.from_metadata(metadata)
    chain.transition(N, kernel_list, target_rows, target_cols)
    return chain.to_metadata()

def _intialize((X, cctypes, distargs, seed)):
    chain = State(X, cctypes, distargs, seed=seed)
    return chain.to_metadata()

class Engine(object):
    """Multiprocessing engine for a stochastic ensemble of parallel StateGPMs."""

    def __init__(self, X, cctypes, distargs, num_states=1, seeds=None,
            metadatas=None, initialize=False):
        """If initialize is True, all metadatas will be resampled!"""
        self.X = X
        self.cctypes = cctypes
        self.distargs = distargs
        self.num_states = num_states
        self.seeds = range(num_states) if seeds is None else seeds
        self.metadata = metadatas
        if initialize:
            self.initialize()

    def initialize(self, multithread=True):
        """Reinitializes all the states from the prior. """
        _, mapper = self._get_mapper(multithread=multithread)
        args = ((self.X, self.cctypes, self.distargs, seed) for
                    seed in self.seeds)
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

    def logpdf(self, query, constraints=None):
        """Predictive probability."""

    def simulate(self, query, constraints=None):
        """Predictive sample."""

    def impute(self, query):
        """Impute data."""

    def dependence_probability(self, col0, col1):
        """Compute dependence probability between `col0, col1` as float."""
        Zvs = [metadata['Zv'] for metadata in self.metadata]
        counts = [Zv[col0]==Zv[col1] for Zv in Zvs]
        return sum(counts) / float(self.num_states)

    def dependence_probability_pairwise(self):
        """Compute dependence probability between all pairs as pd.DataFrame."""
        _, n_cols = self.metadata[0]['X'].shape
        D = np.eye(n_cols)
        for i,j in itertools.combinations(range(n_cols), 2):
            d = self.dependence_probability(i, j)
            D[i,j] = D[j,i] = d
        return D

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

    def to_metadata(self):
        metadata = dict()
        metadata['num_states'] = self.num_states
        metadata['seeds'] = self.seeds
        metadata['metadatas'] = self.metadata
        return metadata

    @classmethod
    def from_metadata(cls, metadata):
        num_states = metadata['num_states']
        seeds = metadata['seeds']
        metadatas = metadata['metadatas']
        return cls(metadatas[0]['X'], metadatas[0]['cctypes'],
            metadatas[0]['distargs'], num_states=num_states, seeds=seeds,
            metadatas=metadatas)

    def to_pickle(self, fileptr):
        metadata = self.to_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_pickle(cls, fileptr):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)
