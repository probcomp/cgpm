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

import math

import numpy as np
from scipy.misc import logsumexp

import gpmcc.utils.sampling as su
import gpmcc.utils.general as utils

def mutual_information_to_linfoot(MI):
    return (1.-math.exp(-2.*MI))**.5

def mutual_information(state, col1, col2, N=1000):
    if state.Zv[col1] != state.Zv[col2]:
        return 0.

    log_crp = su.compute_cluster_crp_logps(state, state.Zv[col1])
    K = len(log_crp)
    clusters_col1 = su.create_clusters(state, col1)
    clusters_col2 = su.create_clusters(state, col2)

    MI = 0
    Px = np.zeros(K)
    Py = np.zeros(K)
    Pxy = np.zeros(K)

    for _ in xrange(N):
        c = utils.log_pflip(log_crp)
        x = clusters_col1[c].simulate()
        y = clusters_col2[c].simulate()
        for k in range(K):
            Px[k] = clusters_col1[k].logpdf(x)
            Py[k] = clusters_col2[k].logpdf(y)
            Pxy[k] = Px[k] + Py[k] + log_crp[k]
            Px[k] += log_crp[k]
            Py[k] += log_crp[k]
        PX = logsumexp(Px)
        PY = logsumexp(Py)
        PXY = logsumexp(Pxy)
        MI += (PXY - PX - PY)

    MI /= float(N)
    if MI < 0.:
        print 'mutual_information: MI < 0 (%f)' % MI
        MI = 0.

    return MI
