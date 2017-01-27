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
import pickle

from collections import namedtuple

import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils import general as gu
from cgpm.utils.parallel_map import parallel_map


# Wrapper for a simple cgpm for optimized dependence_probability.
DummyCgpm = namedtuple('DummyCgpm', ['outputs', 'inputs'])


# Multiprocessing functions.

def _intialize((X, rng, kwargs)):
    state = State(X, rng=rng, **kwargs)
    return state

def _modify((method, state, args)):
    getattr(state, method)(*args)
    return state

def _compose((method, state, cgpm_metadata, args)):
    builder = getattr(
        importlib.import_module(cgpm_metadata['factory'][0]),
        cgpm_metadata['factory'][1])
    cgpm = builder.from_metadata(cgpm_metadata, rng=state.rng)
    getattr(state, method)(cgpm, *args)
    return state

def _evaluate((method, state, args)):
    return getattr(state, method)(*args)


class Engine(object):
    """Multiprocessing engine for a stochastic ensemble of parallel States."""

    def __init__(self, X, num_states=1, rng=None, multiprocess=1, **kwargs):
        mapper = parallel_map if multiprocess else map
        self.rng = gu.gen_rng(1) if rng is None else rng
        X = np.asarray(X)
        args = [(X, rng, kwargs) for rng in self._get_rngs(num_states)]
        self.states = mapper(_intialize, args)

    # --------------------------------------------------------------------------
    # External

    def transition(
            self, N=None, S=None, kernels=None, rowids=None, cols=None,
            views=None, progress=True, checkpoint=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('transition', self.states[i],
                (N, S, kernels, rowids, cols, views, progress, checkpoint))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def transition_lovecat(self, N=None, S=None, kernels=None,
            progress=None, checkpoint=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('transition_lovecat', self.states[i],
                (N, S, kernels, progress, checkpoint))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def transition_loom(self, N=None, S=None, kernels=None,
            progress=None, checkpoint=None, multiprocess=1):
        # Uses Loom multiprocessing rather parallel_map.
        from cgpm.crosscat import loomcat
        loomcat.transition_engine(
            self, N=N, S=S, kernels=kernels, progress=progress,
            checkpoint=checkpoint)

    def transition_foreign(self, N=None, S=None, cols=None, progress=True,
            multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('transition_foreign', self.states[i],
                (N, S, cols, progress))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def incorporate_dim(self, T, outputs, inputs=None, cctype=None,
            distargs=None, v=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('incorporate_dim', self.states[i],
                (T, outputs, inputs, cctype, distargs, v))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def unincorporate_dim(self, col, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('unincorporate_dim', self.states[i],
                (col,))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def incorporate(self, rowid, query, evidence=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('incorporate', self.states[i],
                (rowid, query, evidence))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def unincorporate(self, rowid, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('unincorporate', self.states[i],
                (rowid,))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def update_cctype(self, col, cctype, distargs=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('update_cctype', self.states[i],
                (col, cctype, distargs))
                for i in xrange(self.num_states())]
        self.states = mapper(_modify, args)

    def compose_cgpm(self, cgpms, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('compose_cgpm', self.states[i], cgpms[i].to_metadata(),
                ())
                for i in xrange(self.num_states())]
        self.states = mapper(_compose, args)

    def logpdf(self, rowid, query, evidence=None, accuracy=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('logpdf', self.states[i],
                (rowid, query, evidence, accuracy))
            for i in xrange(self.num_states())]
        logpdfs = mapper(_evaluate, args)
        return logpdfs

    def logpdf_bulk(self, rowids, queries, evidences=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('logpdf_bulk', self.states[i],
                (rowids, queries, evidences))
                for i in xrange(self.num_states())]
        logpdfs = mapper(_evaluate, args)
        return logpdfs

    def logpdf_score(self, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [('logpdf_score', self.states[i],
                ())
                for i in xrange(self.num_states())]
        logpdf_scores = mapper(_evaluate, args)
        return logpdf_scores

    def simulate(self, rowid, query, evidence=None, N=None, accuracy=None,
            multiprocess=1):
        self._seed_states()
        mapper = parallel_map if multiprocess else map
        args = [('simulate', self.states[i],
                (rowid, query, evidence, N, accuracy))
                for i in xrange(self.num_states())]
        samples = mapper(_evaluate, args)
        return samples

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None,
            multiprocess=1):
        """Returns list of simualate_bulk, one for each state."""
        self._seed_states()
        mapper = parallel_map if multiprocess else map
        args = [('simulate_bulk', self.states[i],
                (rowids, queries, evidences, Ns))
                for i in xrange(self.num_states())]
        samples = mapper(_evaluate, args)
        return samples

    def mutual_information(self, col0, col1, evidence=None, T=None, N=None,
            progress=None, multiprocess=1):
        """Returns list of mutual information estimates, one for each state."""
        self._seed_states()
        mapper = parallel_map if multiprocess else map
        args = [('mutual_information', self.states[i],
                (col0, col1, evidence, T, N, progress))
                for i in xrange(self.num_states())]
        mis = mapper(_evaluate, args)
        return mis

    def dependence_probability(self, col0, col1, multiprocess=1):
        """Compute dependence probabilities between col0 and col1."""
        # XXX Ignore multiprocess.
        return [s.dependence_probability(col0, col1) for s in self.states]

    def dependence_probability_pairwise(self):
        """Compute dependence probability between all pairs as matrix."""
        D = np.eye(len(self.states[0].outputs))
        reindex = {c: k for k, c in enumerate(self.states[0].outputs)}
        for i,j in itertools.combinations(self.states[0].outputs, 2):
            d = np.mean(self.dependence_probability(i, j))
            D[reindex[i], reindex[j]] = D[reindex[j], reindex[i]] = d
        return D

    def row_similarity(self, row0, row1, cols=None, multiprocess=1):
        """Compute similiarties between row0 and row1."""
        return [s.row_similarity(row0, row1, cols) for s in self.states]

    def row_similarity_pairwise(self, cols=None):
        """Compute dependence probability between all pairs as matrix."""
        n_rows = self.states[0].n_rows()
        S = np.eye(n_rows)
        for i,j in itertools.combinations(range(n_rows), 2):
            s = np.mean(self.row_similarity(i,j, cols))
            S[i,j] = S[j,i] = s
        return S

    def get_state(self, index):
        return self.states[index]

    def drop_state(self, index):
        del self.states[index]

    def num_states(self):
        return len(self.states)

    # --------------------------------------------------------------------------
    # Internal

    def _seed_states(self):
        rngs = self._get_rngs()
        for rng, state in zip(rngs, self.states):
            state.rng = rng

    def _get_rngs(self, N=None):
        num_states = N if N is not None else self.num_states()
        seeds = self.rng.randint(low=1, high=2**32-1, size=num_states)
        return [gu.gen_rng(s) for s in seeds]

    def _likelihood_weighted_integrate(
            self, logpdfs, rowid, evidence=None, multiprocess=1):
        # Computes an importance sampling integral with likelihood weight.
        assert len(logpdfs) == len(self.states)
        if evidence:
            weights = self.logpdf(rowid, evidence, multiprocess=multiprocess)
            return gu.logmeanexp_weighted(logpdfs, weights)
        else:
            return gu.logmeanexp(logpdfs)

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
        metadata = dict()
        metadata['X'] = self.states[0].data_array().tolist()
        metadata['states'] = [s.to_metadata() for s in self.states]
        for m in metadata['states']:
            del m['X']
        metadata['factory'] = ('cgpm.crosscat.engine', 'Engine')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng=None, multiprocess=1):
        if rng is None:
            rng = gu.gen_rng(0)
        engine = cls(
            X=metadata['X'],
            num_states=0,
            rng=rng,
            multiprocess=multiprocess)
        # Repopulate the states with the dataset.
        for m in metadata['states']:
            m['X'] = metadata['X']
        num_states = len(metadata['states'])
        def retrieve_state((state, rng)):
            return State.from_metadata(state, rng=rng)
        mapper = parallel_map if multiprocess else map
        engine.states = mapper(
            retrieve_state,
            zip(metadata['states'], engine._get_rngs(num_states)))
        return engine

    def to_pickle(self, fileptr):
        metadata = self.to_metadata()
        pickle.dump(metadata, fileptr)

    @classmethod
    def from_pickle(cls, fileptr, rng=None):
        if isinstance(fileptr, str):
            with open(fileptr, 'r') as f:
                metadata = pickle.load(f)
        else:
            metadata = pickle.load(fileptr)
        return cls.from_metadata(metadata, rng=rng)
