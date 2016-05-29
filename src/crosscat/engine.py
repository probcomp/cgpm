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

from scipy.misc import logsumexp

from gpmcc.crosscat.state import State
from gpmcc.utils import general as gu


# Multiprocessing functions.

def _intialize((X, rng, kwargs)):
    state = State(X, rng=rng, **kwargs)
    return state.to_metadata()

def _modify((method, metadata, args)):
    state = State.from_metadata(metadata, rng=metadata['rng'])
    getattr(state, method)(*args)
    return state.to_metadata()

def _evaluate((method, metadata, args)):
    state = State.from_metadata(metadata, rng=metadata['rng'])
    return getattr(state, method)(*args)


class Engine(object):
    """Multiprocessing engine for a stochastic ensemble of parallel States."""

    def __init__(self, X, num_states=1, rng=None, multithread=1, **kwargs):
        self.rng = gu.gen_rng(1) if rng is None else rng
        self.X = np.asarray(X)
        pool, mapper = self._get_mapper(multithread)
        args = ((X, rng, kwargs) for rng in self._get_rngs(num_states))
        self.states = mapper(_intialize, args)
        self._close_mapper(pool)

    # --------------------------------------------------------------------------
    # External

    def transition(self, N=None, S=None, kernels=None, target_views=None,
            target_rows=None, target_cols=None, do_plot=False, do_progress=True,
            multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('transition', self.states[i],
                (N, S, kernels, target_views, target_rows, target_cols,
                    do_plot, do_progress))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def incorporate_dim(self, T, outputs, inputs=None, cctype=None,
            distargs=None, v=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('incorporate_dim', self.states[i],
                (T, outputs, inputs, cctype, distargs, v))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def unincorporate_dim(self, col, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('unincorporate_dim', self.states[i],
                (col,))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def incorporate(self, rowid, query, evidence=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('incorporate', self.states[i],
                (rowid, query, evidence))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def unincorporate(self, rowid, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('unincorporate', self.states[i],
                (rowid,))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def update_cctype(self, col, cctype, distargs=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('update_cctype', self.states[i],
                (col, cctype, distargs))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def logpdf(self, rowid, query, evidence=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('logpdf', self.states[i],
                (rowid, query, evidence))
            for i in xrange(self.num_states())]
        logpdfs = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(logpdfs)

    def logpdf_bulk(self, rowids, queries, evidences=None, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('logpdf_bulk', self.states[i],
                (rowids, queries, evidences))
                for i in xrange(self.num_states())]
        logpdfs = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(logpdfs)

    def logpdf_score(self, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('logpdf_score', self.states[i],
                ())
                for i in xrange(self.num_states())]
        logpdf_scores = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(logpdf_scores)

    def simulate(self, rowid, query, evidence=None, N=1, multithread=1):
        pool, mapper = self._get_mapper(multithread)
        args = [('simulate', self.states[i],
                (rowid, query, evidence, N))
                for i in xrange(self.num_states())]
        samples = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(samples)

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None,
            multithread=1):
        """Returns list of simualate_bulk, one for each state."""
        pool, mapper = self._get_mapper(multithread)
        args = [('simulate_bulk', self.states[i],
                (rowids, queries, evidences, Ns))
                for i in xrange(self.num_states())]
        samples = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(samples)

    def mutual_information(self, col0, col1, evidence=None, N=None,
            multithread=1):
        """Returns list of mutual information estimates, one for each state."""
        pool, mapper = self._get_mapper(multithread)
        args = [('mutual_information', self.states[i],
                (col0, col1, evidence, N))
                for i in xrange(self.num_states())]
        mis = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(mis)

    def conditional_mutual_information(self, col0, col1, evidence, T=None,
            N=None, multithread=1):
        """Returns list of mutual information estimates, one for each state."""
        pool, mapper = self._get_mapper(multithread)
        args = [('conditional_mutual_information', self.states[i],
                (col0, col1, evidence, T, N))
                for i in xrange(self.num_states())]
        mis = mapper(_evaluate, args)
        self._close_mapper(pool)
        return np.asarray(mis)

    def dependence_probability(self, col0, col1, states=None):
        """Compute dependence probability between col0 and col1 as float."""
        if states is None: states = xrange(self.num_states())
        Zvs = [self.states[s]['Zv'] for s in states]
        counts = [Zv[col0]==Zv[col1] for Zv in Zvs]
        return sum(counts) / float(len(states))

    def dependence_probability_pairwise(self, states=None):
        """Compute dependence probability between all pairs as matrix."""
        n_cols = len(self.states[0]['X'][0])
        D = np.eye(n_cols)
        for i,j in itertools.combinations(range(n_cols), 2):
            D[i,j] = D[j,i] = self.dependence_probability(i,j, states=states)
        return D

    def row_similarity(self, row0, row1, cols=None, states=None):
        """Compute similiarty between row0 and row1 as float."""
        if states is None: states = xrange(self.num_states())
        if cols is None: cols = range(len(self.states[0]['cctypes']))
        def row_sim_state(s):
            Zv, Zrv = self.states[s]['Zv'], self.states[s]['Zrv']
            Zrs = [Zrv[v] for v in set(Zv[c] for c in cols)]
            return sum([Zr[row0]==Zr[row1] for Zr in Zrs]) / float(len(Zrv))
        return sum(map(row_sim_state, states)) / len(states)

    def row_similarity_pairwise(self, cols=None, states=None):
        """Compute dependence probability between all pairs as matrix."""
        n_rows = len(self.states[0]['X'])
        S = np.eye(n_rows)
        for i,j in itertools.combinations(range(n_rows), 2):
            S[i,j] = S[j,i] = self.row_similarity(i,j, cols=cols, states=states)
        return S

    def get_state(self, index):
        self._populate_metadata()
        return State.from_metadata(self.states[index])

    def get_states(self, indices):
        self._populate_metadata()
        return [self.get_state(i) for i in indices]

    def drop_state(self, index):
        del self.states[index]

    def drop_states(self, indices):
        drop = set(indices)
        self.states = [m for i,m in enumerate(self.states) if i not in drop]

    def num_states(self):
        return len(self.states)

    # --------------------------------------------------------------------------
    # Internal

    def _get_mapper(self, multithread):
        self._populate_metadata() # XXX Right place?
        pool, mapper = None, map
        if multithread:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            mapper = pool.map
        return pool, mapper

    def _close_mapper(self, pool):
        if pool is not None:
            pool.close()

    def _populate_metadata(self):
        if not hasattr(self, 'states'):
            return
        for rng, state in zip(self._get_rngs(), self.states):
            state['rng'] = rng
            state['X'] = self.X

    def _depopulate_metadata(self):
        if not hasattr(self, 'states'):
            return
        for state in self.states:
            state.pop('rng', None)
            state.pop('X', None)

    def _get_rngs(self, N=None):
        num_states = N if N is not None else self.num_states()
        seeds = self.rng.randint(low=1, high=2**32-1, size=num_states)
        return [gu.gen_rng(s) for s in seeds]

    def _process_logpdfs(self, logpdfs, rowid, evidence=None, multithread=1):
        assert len(logpdfs) == len(self.states)
        weights = np.zeros(len(logpdfs)) if not evidence else\
            self.logpdf(rowid, evidence, evidence=None, multithread=multithread)
        return logsumexp(logpdfs + weights) - logsumexp(weights)

    def _process_samples(self, samples, rowid, evidence=None, multithread=1):
        assert len(samples) == len(self.states)
        assert all(len(s) == len(samples[0]) for s in samples[1:])
        N = len(samples[0])
        weights = np.zeros(len(samples)) if not evidence else\
            self.logpdf(rowid, evidence, multithread=multithread)
        n_model = np.bincount(gu.log_pflip(weights, size=N, rng=self.rng))
        return np.vstack([s[:n] for s,n in zip(samples, n_model) if n])

    # --------------------------------------------------------------------------
    # Serialize

    def to_metadata(self):
        self._depopulate_metadata()
        metadata = dict()
        metadata['X'] = self.X.tolist()
        metadata['states'] = self.states
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None):
        if rng is None:
            rng = gu.gen_rng(0)
        # XXX Backward compatability.
        if 'states' not in metadata:
            metadata['states'] = metadata['state_metadatas']
        if 'X' not in metadata:
            metadata['X'] = metadata['states'][0]['X']
        engine = cls(X=metadata['X'], num_states=0, multihtread=0, rng=rng)
        engine.states = metadata['states']
        engine._populate_metadata()
        return engine

    def to_pickle(self, fileptr):
        metadata = self.to_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_pickle(cls, fileptr, rng=None):
        metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata, rng=rng)
