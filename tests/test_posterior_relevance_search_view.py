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

"""
Find the math for these tests in
https://docs.google.com/document/d/15_JGb39TuuSup_0gIBJTuMHYs8HS4m_TzjJ-AOXnh9M/edit
"""

import numpy as np
from itertools import product
from collections import OrderedDict
from pprint import pprint

from cgpm.mixtures.view import View
from cgpm.utils.general import logsumexp, merged


dpmm_data = [[1, 1, 0, 1, 1, 1],
             [1, 1, 1, 0, 1, 1],
             [1, 1, 1, 1, 0, 1],
             [1, 1, 1, 1, 1, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]]

def initialize_view():
    data = np.array(dpmm_data)
    D = len(data[0])
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    Zr = [0 if i < 4 else 1 for i in range(8)]
    view = View(
        X,
        outputs=[1000] + outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        hypers={
            i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zr=Zr)
    return view

def test_relevance_search_wrt_rows_in_first_cluster():
    view = initialize_view()

    for rowid in xrange(4):
        score = view.posterior_relevance_search(
            query={rowid: {}}, debug=True)

        # Assert highest score with itself
        assert score[0][0] == rowid

        # Assert highest scoring values come from same cluster as query
        first_four = [score[i][0] for i in xrange(4)]
        first_cluster = range(4)
        assert set(first_four) == set(first_cluster)

        # Lowest scoring values come from different cluster than query?
        last_four = [score[i][0] for i in xrange(4, 8)]
        second_cluster = range(4, 8)
        assert set(last_four) == set(second_cluster)

def test_relevance_search_wrt_rows_in_second_cluster():
    view = initialize_view()

    for rowid in xrange(4, 8):
        score = view.posterior_relevance_search(query={rowid: {}}, debug=True)

        # Assert highest score with itself
        assert score[0][0] == rowid

        # Assert highest scoring values come from same cluster as query
        first_four = [score[i][0] for i in xrange(4)]
        second_cluster = range(4, 8)
        assert set(first_four) == set(second_cluster)

        # Assert lowest scoring values come from different cluster than query
        last_four = [score[i][0] for i in xrange(4, 8)]
        first_cluster = range(4)
        assert set(last_four) == set(first_cluster)

def test_relevance_search_mixed():
    view = initialize_view()

    score = view.posterior_relevance_search(query={0: {}, 7: {}}, debug=True)

    # Assert highest scores with itself
    first_two = [score[i][0] for i in xrange(2)]
    assert set(first_two) == {0, 7}

    pprint(score)

def test_relevance_search_wrt_majority():
    view = initialize_view()

    score = view.posterior_relevance_search(query={0: {}, 1: {}, 7: {}}, debug=True)

    # Assert highest scoring values come from majority cluster in query
    first_four = [score[i][0] for i in xrange(4)]
    first_cluster = range(4)
    assert set(first_four) == set(first_cluster)

    # Assert lowest scoring values come from minority cluster in query
    last_four = [score[i][0] for i in xrange(4, 8)]
    second_cluster = range(4, 8)
    assert set(last_four) == set(second_cluster)

    pprint(score)
