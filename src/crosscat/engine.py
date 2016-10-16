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

import importlib
import itertools
import multiprocessing
import pickle

from collections import namedtuple

import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils import general as gu


# Wrapper for a simple cgpm for optimized dependence_probability.
DummyCgpm = namedtuple('DummyCgpm', ['outputs', 'inputs'])


# Multiprocessing functions.

def _intialize((X, rng, kwargs)):
    state = State(X, rng=rng, **kwargs)
    return state.to_metadata()

def _modify((method, metadata, args)):
    state = State.from_metadata(metadata, rng=metadata['rng'])
    getattr(state, method)(*args)
    return state.to_metadata()

def _compose((method, metadata, cgpm_metadata, args)):
    builder = getattr(
        importlib.import_module(cgpm_metadata['factory'][0]),
        cgpm_metadata['factory'][1])
    cgpm = builder.from_metadata(cgpm_metadata, rng=metadata['rng'])
    state = State.from_metadata(metadata, rng=metadata['rng'])
    getattr(state, method)(cgpm, *args)
    return state.to_metadata()

def _evaluate((method, metadata, args)):
    state = State.from_metadata(metadata, rng=metadata['rng'])
    return getattr(state, method)(*args)


class Engine(object):
    """Multiprocessing engine for a stochastic ensemble of parallel States."""

    def __init__(self, X, num_states=1, rng=None, multiprocess=1, **kwargs):
        self.rng = gu.gen_rng(1) if rng is None else rng
        self.X = np.asarray(X)
        pool, mapper = self._get_mapper(multiprocess)
        args = ((X, rng, kwargs) for rng in self._get_rngs(num_states))
        self.states = mapper(_intialize, args)
        self._close_mapper(pool)

    # --------------------------------------------------------------------------
    # External

    def transition(self, N=None, S=None, kernels=None, rowids=None,
            cols=None, views=None, progress=True, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('transition', self.states[i],
                (N, S, kernels, rowids, cols, views, progress))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def transition_lovecat(self, N=None, S=None, kernels=None, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('transition_lovecat', self.states[i],
                (N, S, kernels))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def incorporate_dim(self, T, outputs, inputs=None, cctype=None,
            distargs=None, v=None, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('incorporate_dim', self.states[i],
                (T, outputs, inputs, cctype, distargs, v))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def unincorporate_dim(self, col, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('unincorporate_dim', self.states[i],
                (col,))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def incorporate(self, rowid, query, evidence=None, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('incorporate', self.states[i],
                (rowid, query, evidence))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def unincorporate(self, rowid, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('unincorporate', self.states[i],
                (rowid,))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def update_cctype(self, col, cctype, distargs=None, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('update_cctype', self.states[i],
                (col, cctype, distargs))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)
        self._close_mapper(pool)

    def compose_cgpm(self, cgpms, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('compose_cgpm', self.states[i], cgpms[i].to_metadata(),
                ())
                for i in xrange(self.num_states())]
        self.states = mapper(_compose, args)
        self._close_mapper(pool)

    def logpdf(self, rowid, query, evidence=None, accuracy=None, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('logpdf', self.states[i],
                (rowid, query, evidence, accuracy))
            for i in xrange(self.num_states())]
        logpdfs = mapper(_evaluate, args)
        self._close_mapper(pool)
        return logpdfs

    def logpdf_bulk(self, rowids, queries, evidences=None, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('logpdf_bulk', self.states[i],
                (rowids, queries, evidences))
                for i in xrange(self.num_states())]
        logpdfs = mapper(_evaluate, args)
        self._close_mapper(pool)
        return logpdfs

    def logpdf_score(self, multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('logpdf_score', self.states[i],
                ())
                for i in xrange(self.num_states())]
        logpdf_scores = mapper(_evaluate, args)
        self._close_mapper(pool)
        return logpdf_scores

    def simulate(self, rowid, query, evidence=None, N=None, accuracy=None,
            multiprocess=1):
        pool, mapper = self._get_mapper(multiprocess)
        args = [('simulate', self.states[i],
                (rowid, query, evidence, N, accuracy))
                for i in xrange(self.num_states())]
        samples = mapper(_evaluate, args)
        self._close_mapper(pool)
        return samples

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None,
            multiprocess=1):
        """Returns list of simualate_bulk, one for each state."""
        pool, mapper = self._get_mapper(multiprocess)
        args = [('simulate_bulk', self.states[i],
                (rowids, queries, evidences, Ns))
                for i in xrange(self.num_states())]
        samples = mapper(_evaluate, args)
        self._close_mapper(pool)
        return samples

    def mutual_information(self, col0, col1, evidence=None, N=None,
            multiprocess=1):
        """Returns list of mutual information estimates, one for each state."""
        pool, mapper = self._get_mapper(multiprocess)
        args = [('mutual_information', self.states[i],
                (col0, col1, evidence, N))
                for i in xrange(self.num_states())]
        mis = mapper(_evaluate, args)
        self._close_mapper(pool)
        return mis

    def conditional_mutual_information(self, col0, col1, evidence, T=None,
            N=None, multiprocess=1):
        """Returns list of mutual information estimates, one for each state."""
        pool, mapper = self._get_mapper(multiprocess)
        args = [('conditional_mutual_information', self.states[i],
                (col0, col1, evidence, T, N))
                for i in xrange(self.num_states())]
        mis = mapper(_evaluate, args)
        self._close_mapper(pool)
        return mis

    def dependence_probability(self, col0, col1, multiprocess=1):
        """Compute dependence probability between col0 and col1 as float."""
        # XXX Ignore multiprocess.
        return [
            self._dependence_probability_state(s, col0, col1)
            for s in self.states
        ]

    def _dependence_probability_state(self, state, col0, col1):
        cgpms = [
            DummyCgpm(m['outputs'], m['inputs'])
            for m in state['hooked_cgpms'].itervalues()
        ] + [DummyCgpm(state['outputs'], [])]
        return State._dependence_probability(
            cgpms, dict(state['Zv']), col0, col1)

    def dependence_probability_pairwise(self):
        """Compute dependence probability between all pairs as matrix."""
        n_cols = len(self.states[0]['X'][0])
        D = np.eye(n_cols)
        for i,j in itertools.combinations(range(n_cols), 2):
            d = np.mean(self.dependence_probability(i,j))
            D[i,j] = D[j,i] = d
        return D

    def row_similarity(
            self, row0, row1, cols=None, states=None, multiprocess=1):
        """Compute similiarty between row0 and row1 as float."""
        if states is None: states = xrange(self.num_states())
        if cols is None: cols = range(len(self.states[0]['cctypes']))
        def row_sim_state(s):
            Zv = dict(self.states[s]['Zv'])
            Zrv = dict(self.states[s]['Zrv'])
            Zrs = [Zrv[v] for v in set(Zv[c] for c in cols)]
            return sum([Zr[row0]==Zr[row1] for Zr in Zrs]) / float(len(Zrv))
        return [row_sim_state(s) for s in states]

    def row_similarity_pairwise(self, cols=None, states=None):
        """Compute dependence probability between all pairs as matrix."""
        n_rows = len(self.states[0]['X'])
        S = np.eye(n_rows)
        for i,j in itertools.combinations(range(n_rows), 2):
            s = np.mean(self.row_similarity(i,j, cols=cols, states=states))
            S[i,j] = S[j,i] = s
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

    def _get_mapper(self, multiprocess):
        self._populate_metadata() # XXX Right place?
        pool, mapper = None, map
        if multiprocess:
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

    def _likelihood_weighted_integrate(
            self, logpdfs, rowid, evidence=None, multiprocess=1):
        # Computes an importance sampling integral with likelihood weight.
        assert len(logpdfs) == len(self.states)
        weights = np.zeros(len(logpdfs)) if not evidence else\
            self.logpdf(rowid, evidence, evidence=None, multiprocess=multiprocess)
        return gu.logsumexp(logpdfs + weights) - gu.logsumexp(weights)

    def _likelihood_weighted_resample(
            self, samples, rowid, evidence=None, multiprocess=1):
        assert len(samples) == len(self.states)
        assert all(len(s) == len(samples[0]) for s in samples[1:])
        N = len(samples[0])
        weights = np.zeros(len(samples)) if not evidence else\
            self.logpdf(rowid, evidence, multiprocess=multiprocess)
        n_model = np.bincount(gu.log_pflip(weights, size=N, rng=self.rng))
        resamples = [s[:n] for s,n in zip(samples, n_model) if n]
        return list(itertools.chain.from_iterable(resamples))

    # --------------------------------------------------------------------------
    # Serialize

    def to_metadata(self):
        self._depopulate_metadata()
        metadata = dict()
        metadata['X'] = self.X.tolist()
        metadata['states'] = self.states
        metadata['factory'] = ('cgpm.crosscat.engine', 'Engine')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None, multiprocess=1):
        if rng is None:
            rng = gu.gen_rng(0)
        # XXX Backward compatability.
        if 'states' not in metadata:
            metadata['states'] = metadata['state_metadatas']
        if 'X' not in metadata:
            metadata['X'] = metadata['states'][0]['X']
        engine = cls(
            X=metadata['X'],
            num_states=0,
            rng=rng,
            multiprocess=multiprocess)
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
