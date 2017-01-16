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

def test_relevance_score_ordering_wrt_row_3():
    view = initialize_view()

    score = []
    # import pudb; pudb.set_trace()
    for row in xrange(len(view.X.values()[0])):  # for each row in the dataset
        score.append(view.relevance_score(
            query={row: {}}, evidence={3: {}},
            debug=True))
    
    pprint(score)

    # Assert equality between scores of rows 4 to 7
    assert len(set(score[4::7])) == 1

    # Assert equality between scores of rows 0 to 2
    assert len(set(score[0::2])) == 1

    # Assert score(row 3; row 3) > score(row 2; row 3)
    assert score[3] > score[2]

    # Assert score(row 2; row 3) > score(row 4; row 3)
    assert score[2] > score[4]

def test_relevance_search_wrt_row_3():
    view = initialize_view()

    sorted_score = view.relevance_search(evidence={3: {}})
    from pprint import pprint

    pprint(sorted_score)

# def test_relevance_score_of_fourth_row_wrt_fourth_row():
#     """
#     TODO: assert each step below with logpdf_set
#     """

#     view = initialize_view()
    
#     # Analytically computed score
#     # 1.1 logp(row4, row4*| z4*=z4=0)
#     logp_H1_z0 = np.log(
#         5**2 * 4**3 * 2 * 4**2 * 3**3 * 1) - 6*np.log(30)
#     # 1.2 logp(row4, row4*| z4*=z4=1)
#     logp_H1_z1 = np.log(
#         3**4 * 2 * 5 * 2**4 * 1 * 4) - 6*np.log(30)
#     logp_H1_z2 = np.log(
#         3**4 * 2 * 5 * 2**4 * 1 * 4) - 6*np.log(30)
#     # 1.4 logp(row4, row4*| H1)
#     logp_H1 = logsumexp([logp_H1_z0, logp_H1_z1])

#     # 2.1 logp(row4, row4*| z4*=1, z4=0)
#     logp_H2_z1z0 = np.log(
#         2**4 * 1 * 4 * 4**2 * 3**3 * 1) - 6*np.log(25)
#     # 1.2 logp(row4, row4*| z4*=z4=0)
#     logp_H2_z0z1 = np.log(
#         4**2 * 3**3 * 1 * 2**4 * 1 * 4) - 6*np.log(25)
#     # 1.3 logp(row4, row4*| H1)
#     logp_H2 = logsumexp((logp_H2_z0z1, logp_H2_z1z0))

#     math_out = logp_H1 - logp_H2
#     # import pudb; pudb.set_trace()
#     test_out = view.relevance_score(query={4: {}}, evidence={4: {}})

#     assert np.allclose(math_out, test_out)
