# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import gpmcc.utils.sampling as su
import gpmcc.utils.general as utils

from scipy.misc import logsumexp

import math
import numpy

def mutual_information_to_linfoot(MI):
    return ( 1.0 - math.exp(-2.0 * MI)) ** 0.5

def mutual_information(state, col_1, col_2, N=1000):
    view_1 = state.Zv[col_1]
    view_2 = state.Zv[col_2]

    if view_1 != view_2:
        print("mutual_information: not in same view: MI = 0.0")
        return 0.0

    log_crp = su.get_cluster_crps(state, view_1)
    K = len(log_crp)

    clusters_col_1 = su.create_cluster_set(state, col_1)
    clusters_col_2 = su.create_cluster_set(state, col_2)

    MI = 0

    Px = numpy.zeros(K)
    Py = numpy.zeros(K)
    Pxy = numpy.zeros(K)

    for i in range(N):
        c = utils.log_pflip(log_crp)
        x = clusters_col_1[c].predictive_draw()
        y = clusters_col_2[c].predictive_draw()
        for k in range(K):
            Px[k] = clusters_col_1[k].predictive_logp(x)
            Py[k] = clusters_col_2[k].predictive_logp(y)
            Pxy[k] = Px[k] + Py[k] + log_crp[k]
            Px[k] += log_crp[k]
            Py[k] += log_crp[k]
        PX = logsumexp(Px)
        PY = logsumexp(Py)
        PXY = logsumexp(Pxy)
        MI += (PXY - PX - PY)

    MI /= float(N)
    if MI < 0.0:
        print("mutual_information: MI < 0 (%f)" % MI)
        MI = 0.0

    return MI
