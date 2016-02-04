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

# XXX Multiprocessing functions.

def _intialize((X, cctypes, distargs, seed)):
    state = State(X, cctypes, distargs, seed=seed)
    return state.to_metadata()

def _transition((metadata, N, kernels, target_views, target_rows, target_cols)):
    state = State.from_metadata(metadata)
    state.transition(N=N, kernels=kernels, target_views=target_views,
        target_rows=target_rows, target_cols=target_cols)
    return state.to_metadata()

def _incorporate_dim((metadata, X, cctype, distargs, v)):
    chain = State.from_metadata(metadata)
    chain.incorporate_dim(X, cctype, distargs=distargs, v=v)
    return chain.to_metadata()

def _unincorporate_dim((metadata, col)):
    state = State.from_metadata(metadata)
    state.unincorporate_dim(col)
    return state.to_metadata()

def _incorporate_rows((metadata, X, k)):
    state = State.from_metadata(metadata)
    state.incorporate_rows(X, k=k)
    return state.to_metadata()

def _unincorporate_rows((metadata, rowid)):
    state = State.from_metadata(metadata)
    state.unincorporate_rows(rowid)
    return state.to_metadata()

def _logpdf((metadata, rowid, query, evidence)):
    state = State.from_metadata(metadata)
    return state.logpdf(rowid, query, evidence=evidence)

def _logpdf_marginal((metadata)):
    state = State.from_metadata(metadata)
    return state.logpdf_marginal()

def _simulate((metadata, rowid, query, evidence, N)):
    state = State.from_metadata(metadata)
    return state.simulate(rowid, query, evidence=evidence, N=N)

# XXX Multiprocessing functions.

class Engine(object):
    """Multiprocessing engine for a stochastic ensemble of parallel States."""

    def __init__(self, X, cctypes, distargs, num_states=1, seeds=None,
            state_metadatas=None, initialize=False):
        """If initialize is True all state_metadatas will be resampled!"""
        self.X = X
        self.cctypes = cctypes
        self.distargs = distargs
        self.num_states = num_states
        self.seeds = range(num_states) if seeds is None else seeds
        self.metadata = state_metadatas
        if initialize:
            self.initialize()

    def initialize(self, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = ((self.X, self.cctypes, self.distargs, seed) for seed in
            self.seeds)
        self.metadata = mapper(_intialize, args)

    def transition(self, N=1, kernels=None, target_views=None, target_rows=None,
            target_cols=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], N, kernels, target_views, target_rows,
            target_cols) for i in xrange(self.num_states)]
        self.metadata = mapper(_transition, args)

    def incorporate_dim(self, X, cctype, distargs=None, v=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], X, cctype, distargs, v) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_incorporate_dim, args)

    def unincorporate_dim(self, col, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], col) for i in xrange(self.num_states)]
        self.metadata = mapper(_unincorporate_dim, args)

    def incorporate_rows(self, X, k=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], X, k) for i in xrange(self.num_states)]
        self.metadata = mapper(_incorporate_rows, args)

    def unincorporate_rows(self, rowid, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], rowid) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_unincorporate_rows, args)

    def logpdf(self, rowid, query, evidence=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], rowid, query, evidence) for i in
            xrange(self.num_states)]
        logpdfs = mapper(_logpdf, args)
        return logpdfs

    def logpdf_marginal(self, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i]) for i in xrange(self.num_states)]
        logpdf_marginals = mapper(_logpdf_marginal, args)
        return logpdf_marginals

    def simulate(self, rowid, query, evidence=None, N=1, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [(self.metadata[i], rowid, query, evidence, N) for i in
            xrange(self.num_states)]
        samples = mapper(_simulate, args)
        return samples

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

    def row_similarity(self,row0, row1):
        """Compute similiarty between row0 and row1 as float."""
        prob = 0
        for Zrv in [metadata['Zrcv'] for metadata in self.metadata]:
            prob += sum([Zr[row0]==Zr[row1] for Zr in Zrv]) / float(len(Zrv))
        return prob / self.num_states

    def row_similarity_pairwise(self):
        """Compute dependence probability between all pairs as matrix."""
        n_rows, _ = self.metadata[0]['X'].shape
        S = np.eye(n_rows)
        for i,j in itertools.combinations(range(n_rows), 2):
            S[i,j] = S[j,i] = self.row_similarity(i,j)
        return S

    def get_state(self, index):
        return State.from_metadata(self.metadata[index])

    def _get_mapper(self, multithread):
        pool, mapper = None, map
        if multithread:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            mapper = pool.map
        return pool, mapper

    def to_metadata(self):
        metadata = dict()
        metadata['num_states'] = self.num_states
        metadata['seeds'] = self.seeds
        metadata['state_metadatas'] = self.metadata
        return metadata

    @classmethod
    def from_metadata(cls, metadata):
        return cls(metadata['state_metadatas'][0]['X'],
            metadata['state_metadatas'][0]['cctypes'],
            metadata['state_metadatas'][0]['distargs'],
            num_states=metadata['num_states'],
            seeds=metadata['seeds'],
            state_metadatas=metadata['state_metadatas'])

    def to_pickle(self, fileptr):
        metadata = self.to_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_pickle(cls, fileptr):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata)
