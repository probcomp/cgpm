# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

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
            Px[k] = clusters_col1[k].predictive_logp(x)
            Py[k] = clusters_col2[k].predictive_logp(y)
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
