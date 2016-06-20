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

from cgpm.utils import validation as vu


def test_partition_query_evidence_dict():
    Zv = [0,0,0,1,1,1,2,2,2,3]
    query = {9:101, 1:1, 4:2, 5:7, 7:0}
    evidence = {2:4, 3:1, 9:1, 6:-1, 0:0}
    queries, evidences = vu.partition_query_evidence(Zv, query, evidence)

    # All 4 views have query.
    assert len(queries) == 4

    # View 0 has 2 queries.
    assert len(queries[0]) == 1
    assert queries[0][1] == 1
    # View 1 has 2 queries.
    assert len(queries[1]) == 2
    assert queries[1][4] == 2
    assert queries[1][5] == 7
    # View 2 has 1 queries.
    assert len(queries[2]) == 1
    assert queries[2][7] == 0
    # View 3 has 1 queries.
    assert len(queries[3]) == 1
    assert queries[3][9] == 101

    # Views 0,1,2,3 have evidence.
    assert len(evidences) == 4
    # View 0 has 2 evidence.
    assert len(evidences[0]) == 2
    assert evidences[0][0] == 0
    assert evidences[0][2] == 4
    # View 1 has 1 evidence.
    assert len(evidences[1]) == 1
    assert evidences[1][3] == 1
    # View 2 has 1 evidence.
    assert len(evidences[2]) == 1
    assert evidences[2][6] == -1
    # View 3 has 1 evidence.
    assert len(evidences[3]) == 1
    assert evidences[3][9] == 1


def test_partition_query_evidence_list():
    Zv = [0,0,0,1,1,1,2,2,2,3]
    query = [9, 1, 4, 5, 7]
    evidence = {2:-4, 3:-1, 9:-1, 6:1, 0:100}
    queries, evidences = vu.partition_query_evidence(Zv, query, evidence)

    # All 4 views have query.
    assert len(queries) == 4

    # View 0 has 2 queries.
    assert len(queries[0]) == 1
    assert 1 in queries[0]
    # View 1 has 2 queries.
    assert len(queries[1]) == 2
    assert 4 in queries[1]
    assert 5 in queries[1]
    # View 2 has 1 queries.
    assert len(queries[2]) == 1
    assert 7 in queries[2]
    # View 3 has 1 queries.
    assert len(queries[3]) == 1
    assert 9 in queries[3]

    # Views 0,1,2,3 have evidence.
    assert len(evidences) == 4
    # View 0 has 2 evidence.
    assert len(evidences[0]) == 2
    assert evidences[0][0] == 100
    assert evidences[0][2] == -4
    # View 1 has 1 evidence.
    assert len(evidences[1]) == 1
    assert evidences[1][3] == -1
    # View 2 has 1 evidence.
    assert len(evidences[2]) == 1
    assert evidences[2][6] == 1
    # View 3 has 1 evidence.
    assert len(evidences[3]) == 1
    assert evidences[3][9] == -1
