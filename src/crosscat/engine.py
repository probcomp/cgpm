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

def _intialize((X, seed, kwargs)):
    state = State(X, rng=gu.gen_rng(seed), **kwargs)
    return state

def _modify((method, state, args)):
    getattr(state, method)(*args)
    return state

def _alter((funcs, state)):
    for func in funcs:
        state = func(state)
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
        args = [(X, seed, kwargs) for seed in self._get_seeds(num_states)]
        self.states = mapper(_intialize, args)

    # --------------------------------------------------------------------------
    # External

    def transition(
            self, N=None, S=None, kernels=None, rowids=None, cols=None,
            views=None, progress=True, checkpoint=None, statenos=None,
            multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('transition', self.states[s],
                (N, S, kernels, rowids, cols, views, progress, checkpoint))
                for s in statenos]
        states = mapper(_modify, args)
        for s, state in zip(statenos, states):
            self.states[s] = state

    def transition_lovecat(
            self, N=None, S=None, kernels=None, rowids=None,
            cols=None, progress=None, checkpoint=None, statenos=None,
            multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('transition_lovecat', self.states[s],
                (N, S, kernels, rowids, cols, progress, checkpoint))
                for s in statenos]
        states = mapper(_modify, args)
        for s, state in zip(statenos, states):
            self.states[s] = state

    def transition_loom(self, N=None, S=None, kernels=None,
            progress=None, checkpoint=None, multiprocess=1):
        # Uses Loom multiprocessing rather parallel_map.
        from cgpm.crosscat import loomcat
        loomcat.transition_engine(
            self, N=N, S=S, kernels=kernels, progress=progress,
            checkpoint=checkpoint)

    def transition_foreign(self, N=None, S=None, cols=None, progress=True,
            statenos=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('transition_foreign', self.states[s],
                (N, S, cols, progress))
                for s in statenos]
        states = mapper(_modify, args)
        for s, state in zip(statenos, states):
            self.states[s] = state

    def incorporate_dim(self, T, outputs, inputs=None, cctype=None,
            distargs=None, v=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('incorporate_dim', self.states[s],
                (T, outputs, inputs, cctype, distargs, v))
                for s in statenos]
        self.states = mapper(_modify, args)

    def unincorporate_dim(self, col, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('unincorporate_dim', self.states[s],
                (col,))
                for s in statenos]
        self.states = mapper(_modify, args)

    def incorporate(self, rowid, query, evidence=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('incorporate', self.states[s],
                (rowid, query, evidence))
                for s in statenos]
        self.states = mapper(_modify, args)

    def unincorporate(self, rowid, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('unincorporate', self.states[s],
                (rowid,))
                for s in statenos]
        self.states = mapper(_modify, args)

    def force_cell(self, rowid, query, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('force_cell', self.states[s],
                (rowid, query, evidence))
                for s in statenos]
        self.states = mapper(_modify, args)

    def force_cell_bulk(self, rowids, queries, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('force_cell_bulk', self.states[s],
                (rowids, queries))
                for s in statenos]
        self.states = mapper(_modify, args)

    def update_cctype(self, col, cctype, distargs=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('update_cctype', self.states[s],
                (col, cctype, distargs))
                for s in statenos]
        self.states = mapper(_modify, args)

    def compose_cgpm(self, cgpms, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = xrange(self.num_states())
        args = [('compose_cgpm', self.states[s], cgpms[s].to_metadata(),
                ())
                for s in statenos]
        self.states = mapper(_compose, args)

    def logpdf(self, rowid, query, evidence=None, accuracy=None,
            statenos=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('logpdf', self.states[s],
                (rowid, query, evidence, accuracy))
            for s in statenos]
        logpdfs = mapper(_evaluate, args)
        return logpdfs

    def logpdf_bulk(self, rowids, queries, evidences=None, statenos=None,
            multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('logpdf_bulk', self.states[s],
                (rowids, queries, evidences))
                for s in statenos]
        logpdfs = mapper(_evaluate, args)
        return logpdfs

    def logpdf_score(self, statenos=None, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('logpdf_score', self.states[s],
                ())
                for s in statenos]
        logpdf_scores = mapper(_evaluate, args)
        return logpdf_scores

    def simulate(self, rowid, query, evidence=None, N=None, accuracy=None,
            statenos=None, multiprocess=1):
        self._seed_states()
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('simulate', self.states[s],
                (rowid, query, evidence, N, accuracy))
                for s in statenos]
        samples = mapper(_evaluate, args)
        return samples

    def simulate_bulk(self, rowids, queries, evidences=None, Ns=None,
            statenos=None, multiprocess=1):
        """Returns list of simualate_bulk, one for each state."""
        self._seed_states()
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('simulate_bulk', self.states[s],
                (rowids, queries, evidences, Ns))
                for s in statenos]
        samples = mapper(_evaluate, args)
        return samples

    def mutual_information(self, col0, col1, evidence=None, T=None, N=None,
            progress=None, statenos=None, multiprocess=1):
        """Returns list of mutual information estimates, one for each state."""
        self._seed_states()
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('mutual_information', self.states[s],
                (col0, col1, evidence, T, N, progress))
                for s in statenos]
        mis = mapper(_evaluate, args)
        return mis

    def dependence_probability(self, col0, col1, statenos=None, multiprocess=1):
        """Compute dependence probabilities between col0 and col1."""
        # XXX Ignore multiprocess.
        statenos = statenos or xrange(self.num_states())
        return [self.states[s].dependence_probability(col0, col1)
            for s in statenos]

    def dependence_probability_pairwise(self, colnos=None, statenos=None,
            multiprocess=1):
        """Compute dependence probability between all pairs as matrix."""
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('dependence_probability_pairwise', self.states[s],
                (colnos,))
                for s in statenos]
        Ds = mapper(_evaluate, args)
        return Ds

    def row_similarity(self, row0, row1, cols=None, statenos=None,
            multiprocess=1):
        """Compute similarities between row0 and row1."""
        statenos = statenos or xrange(self.num_states())
        # XXX Ignore multiprocess.
        return [self.states[s].row_similarity(row0, row1, cols)
            for s in statenos]

    def relevance_probability(
            self, rowid_target, rowid_query, col, hypotheticals=None,
            statenos=None, multiprocess=1):
        """Compute relevance probability of query rows for target row."""
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [('relevance_probability', self.states[s],
                (rowid_target, rowid_query, col, hypotheticals))
                for s in statenos]
        probs = mapper(_evaluate, args)
        return probs

    def row_similarity_pairwise(self, cols=None, statenos=None):
        """Compute dependence probability between all pairs as matrix."""
        n_rows = self.states[0].n_rows()
        S = np.eye(n_rows)
        for i,j in itertools.combinations(range(n_rows), 2):
            s = np.mean(self.row_similarity(i,j, cols, statenos=statenos))
            S[i,j] = S[j,i] = s
        return S

    def alter(self, funcs, statenos=None, multiprocess=1):
        """Apply generic funcs on states in parallel."""
        mapper = parallel_map if multiprocess else map
        statenos = statenos or xrange(self.num_states())
        args = [(funcs, self.states[s]) for s in statenos]
        states = mapper(_alter, args)
        for s, state in zip(statenos, states):
            self.states[s] = state

    def get_state(self, index):
        return self.states[index]

    def drop_state(self, index):
        del self.states[index]

    def num_states(self):
        return len(self.states)

    def add_state(self, count=1, multiprocess=1, **kwargs):
        mapper = parallel_map if multiprocess else map
        # XXX Temporarily disallow adding states for composite CGPM.
        if self.states[0].is_composite():
            raise ValueError('Cannot add new states to composite CGPMs.')
        # Arguments must be the same for all states.
        forbidden = [ 'X', 'outputs', 'cctypes', 'distargs']
        if [f for f in forbidden if f in kwargs]:
            raise ValueError('Cannot specify arguments for: %s.' % (forbidden,))
        X = self.states[0].data_array()
        kwargs['cctypes'] = self.states[0].cctypes()
        kwargs['distargs'] = self.states[0].distargs()
        kwargs['outputs'] = self.states[0].outputs
        args = [(X, seed, kwargs) for seed in self._get_seeds(count)]
        new_states = mapper(_intialize, args)
        self.states.extend(new_states)


    # --------------------------------------------------------------------------
    # Internal

    def _seed_states(self):
        seeds = self._get_seeds()
        for seed, state in zip(seeds, self.states):
            state.rng.seed(seed)

    def _get_seeds(self, N=None):
        num_draws = N if N is not None else self.num_states()
        return self.rng.randint(low=1, high=2**32-1, size=num_draws)

    def _likelihood_weighted_integrate(
            self, logpdfs, rowid, evidence=None, statenos=None, multiprocess=1):
        # Computes an importance sampling integral with likelihood weight.
        assert len(logpdfs) == (
            len(self.states) if statenos is None else len(statenos))
        if evidence:
            weights = self.logpdf(
                rowid, evidence, statenos=statenos, multiprocess=multiprocess)
            return gu.logmeanexp_weighted(logpdfs, weights)
        else:
            return gu.logmeanexp(logpdfs)

    def _likelihood_weighted_resample(
            self, samples, rowid, evidence=None, statenos=None, multiprocess=1):
        assert len(samples) == (
            len(self.states) if statenos is None else len(statenos))
        assert all(len(s) == len(samples[0]) for s in samples[1:])
        N = len(samples[0])
        weights = np.zeros(len(samples)) if not evidence else\
            self.logpdf(
                rowid, evidence, statenos=statenos, multiprocess=multiprocess)
        n_model = np.bincount(gu.log_pflip(weights, size=N, rng=self.rng))
        indexes = [self.rng.choice(N, size=n, replace=False) for n in n_model]
        resamples = [
            [s[i] for i in index]
            for s, index in zip(samples, indexes)
            if len(index) > 0
        ]
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
        def retrieve_state((state, seed)):
            return State.from_metadata(state, rng=gu.gen_rng(seed))
        mapper = parallel_map if multiprocess else map
        engine.states = mapper(
            retrieve_state,
            zip(metadata['states'], engine._get_seeds(num_states)))
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
