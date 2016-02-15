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

# Multiprocessing functions.

def _intialize((X, cctypes, distargs, seed)):
    state = State(X, cctypes, distargs=distargs, seed=seed)
    return state.to_metadata()

def _modify((name, metadata, args)):
    state = State.from_metadata(metadata)
    getattr(state, name)(*args)
    return state.to_metadata()

def _evaluate((name, metadata, args)):
    state = State.from_metadata(metadata)
    return getattr(state, name)(*args)

class Engine(object):
    """Multiprocessing engine for a stochastic ensemble of parallel States."""

    def __init__(self, X, cctypes, distargs=None, num_states=1, seeds=None,
            state_metadatas=None, initialize=False):
        """Do not explicitly use state_metadatas, use Engine.from_metadata."""
        self._X , self._cctypes, self._distargs = X, cctypes, distargs
        self.num_states = num_states
        self.seeds = range(num_states) if seeds is None else seeds
        self.metadata = state_metadatas
        if initialize:
            self.initialize()

    def initialize(self, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = ((self._X, self._cctypes, self._distargs, seed) for seed in
            self.seeds)
        self.metadata = mapper(_intialize, args)
        del (self._X, self._cctypes, self._distargs)

    def transition(self, N=1, S=None, kernels=None, target_views=None,
            target_rows=None, target_cols=None, do_plot=False, do_progress=True,
            multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('transition', self.metadata[i], (N, S, kernels, target_views,
            target_rows, target_cols, do_plot, do_progress)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)

    def incorporate_dim(self, X, cctype, distargs=None, v=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('incorporate_dim', self.metadata[i],
            (X, cctype, distargs, v)) for i in xrange(self.num_states)]
        self.metadata = mapper(_modify, args)

    def unincorporate_dim(self, col, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('unincorporate_dim', self.metadata[i], (col,)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)

    def incorporate_rows(self, X, k=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('incorporate_rows', self.metadata[i], (X, k)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)

    def unincorporate_rows(self, rowid, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('unincorporate_rows', self.metadata[i], (rowid,)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)

    def logpdf(self, rowid, query, evidence=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('logpdf', self.metadata[i], (rowid, query, evidence)) for i in
            xrange(self.num_states)]
        logpdfs = mapper(_evaluate, args)
        return logpdfs

    def logpdf_bulk(self, rowids, queries, evidences=None, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('logpdf_bulk', self.metadata[i], (rowids, queries, evidences))
            for i in xrange(self.num_states)]
        logpdfs = mapper(_evaluate, args)
        return logpdfs

    def logpdf_marginal(self, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('logpdf_marginal', self.metadata[i], ()) for i in
            xrange(self.num_states)]
        logpdf_marginals = mapper(_evaluate, args)
        return logpdf_marginals

    def simulate(self, rowid, query, evidence=None, N=1, multithread=1):
        _, mapper = self._get_mapper(multithread)
        args = [('simulate', self.metadata[i], (rowid, query, evidence, N)) for
            i in xrange(self.num_states)]
        samples = mapper(_evaluate, args)
        return np.asarray(samples)

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None,
            multithread=1):
        """Returns list of simualate_bulk, one for each state."""
        _, mapper = self._get_mapper(multithread)
        args = [('simulate_bulk', self.metadata[i],
            (rowids, queries, evidences, Ns)) for i in xrange(self.num_states)]
        samples = mapper(_evaluate, args)
        return samples

    def dependence_probability(self, col0, col1, states=None):
        """Compute dependence probability between col0 and col1 as float."""
        if states is None:
            states = xrange(self.num_states)
        Zvs = [self.metadata[s]['Zv'] for s in states]
        counts = [Zv[col0]==Zv[col1] for Zv in Zvs]
        return sum(counts) / float(len(states))

    def dependence_probability_pairwise(self, states=None):
        """Compute dependence probability between all pairs as matrix."""
        n_cols = len(self.metadata[0]['X'][0])
        D = np.eye(n_cols)
        for i,j in itertools.combinations(range(n_cols), 2):
            D[i,j] = D[j,i] = self.dependence_probability(i,j, states=states)
        return D

    def row_similarity(self,row0, row1, states=None):
        """Compute similiarty between row0 and row1 as float."""
        if states is None:
            states = xrange(self.num_states)
        prob = 0
        for Zrv in [self.metadata[s]['Zrcv'] for s in states]:
            prob += sum([Zr[row0]==Zr[row1] for Zr in Zrv]) / float(len(Zrv))
        return prob / len(states)

    def row_similarity_pairwise(self, states=None):
        """Compute dependence probability between all pairs as matrix."""
        n_rows = len(self.metadata[0]['X'])
        S = np.eye(n_rows)
        for i,j in itertools.combinations(range(n_rows), 2):
            S[i,j] = S[j,i] = self.row_similarity(i,j, states=states)
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
