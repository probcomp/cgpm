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
        pool, mapper = self._get_mapper(multithread)
        args = ((self._X, self._cctypes, self._distargs, seed) for seed in
            self.seeds)
        self.metadata = mapper(_intialize, args)
        self._close_mapper(pool)
        del (self._X, self._cctypes, self._distargs)

    def transition(self, N=None, S=None, kernels=None, target_views=None,
            target_rows=None, target_cols=None, do_plot=False, do_progress=True,
            multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('transition', self.metadata[i], (N, S, kernels, target_views,
            target_rows, target_cols, do_plot, do_progress)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)
        self._close_mapper(pool)

    def incorporate_dim(self, X, cctype, distargs=None, v=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('incorporate_dim', self.metadata[i],
            (X, cctype, distargs, v)) for i in xrange(self.num_states)]
        self.metadata = mapper(_modify, args)
        self._close_mapper(pool)

    def unincorporate_dim(self, col, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('unincorporate_dim', self.metadata[i], (col,)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)
        self._close_mapper(pool)

    def incorporate_rows(self, X, k=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('incorporate_rows', self.metadata[i], (X, k)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)
        self._close_mapper(pool)

    def unincorporate_rows(self, rowid, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('unincorporate_rows', self.metadata[i], (rowid,)) for i in
            xrange(self.num_states)]
        self.metadata = mapper(_modify, args)
        self._close_mapper(pool)

    def logpdf(self, rowid, query, evidence=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('logpdf', self.metadata[i], (rowid, query, evidence)) for i in
            xrange(self.num_states)]
        logpdfs = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(logpdfs)

    def logpdf_bulk(self, rowids, queries, evidences=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('logpdf_bulk', self.metadata[i], (rowids, queries, evidences))
            for i in xrange(self.num_states)]
        logpdfs = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(logpdfs)

    def logpdf_marginal(self, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('logpdf_marginal', self.metadata[i], ()) for i in
            xrange(self.num_states)]
        logpdf_marginals = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(logpdf_marginals)

    def simulate(self, rowid, query, evidence=None, N=1, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('simulate', self.metadata[i], (rowid, query, evidence, N)) for
            i in xrange(self.num_states)]
        samples = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(samples)

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None,
            multithread=1):
        """Returns list of simualate_bulk, one for each state."""
        pool, mapper = self._get_mapper(multithread)
        args = [('simulate_bulk', self.metadata[i],
            (rowids, queries, evidences, Ns)) for i in xrange(self.num_states)]
        samples = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(samples)

    def mutual_information(self, col0, col1, evidence=None, N=1000,
            multithread=1):
        """Returns list of mutual information estimates, one for each state."""
        pool, mapper = self._get_mapper(multithread)
        args = [('mutual_information', self.metadata[i],
            (col0, col1, evidence, N)) for i in xrange(self.num_states)]
        mis = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(mis)

    def conditional_mutual_information(self, col0, col1, evidence, T=100,
            N=1000, multithread=1):
        """Returns list of mutual information estimates, one for each state."""
        pool, mapper = self._get_mapper(multithread)
        args = [('conditional_mutual_information', self.metadata[i],
            (col0, col1, evidence, T, N)) for i in xrange(self.num_states)]
        mis = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(mis)

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
        for Zrv in [self.metadata[s]['Zrv'] for s in states]:
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

    def get_states(self, indices):
        return [self.get_state(i) for i in indices]

    def drop_state(self, index):
        del self.metadata[index]
        self.num_states = len(self.metadata)

    def drop_states(self, indices):
        drop = set(indices)
        self.metadata = [m for i,m in enumerate(self.metadata) if i not in drop]
        self.num_states = len(self.metadata)

    def _get_mapper(self, multithread):
        pool, mapper = None, map
        if multithread:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            mapper = pool.map
        return pool, mapper

    def _close_mapper(self, pool):
        if pool is not None:
            pool.close()

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
