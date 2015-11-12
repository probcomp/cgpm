import gpmcc.utils.sampling as su
import gpmcc.utils.general as utils

from scipy.misc import logsumexp

import math
import numpy

def mutual_information_to_linfoot(MI):
    return (1.0-math.exp(-2.0*MI))**0.5

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
            Pxy[k] = Px[k]+Py[k]+log_crp[k]
            Px[k] += log_crp[k]
            Py[k] += log_crp[k]

        PX = logsumexp(Px)
        PY = logsumexp(Py)
        PXY = logsumexp(Pxy)

        MI += (PXY-PX-PY)

    MI /= float(N)

    if MI < 0.0:
        print("mutual_information: MI < 0 (%f)" % MI)
        MI = 0.0

    return MI

def test_predictive_draw(state, N=None):
    import pylab
    if state.n_cols != 2:
        print("state must have exactly 2 columns")
        return

    if N is None:
        N = state.n_rows

    view_1 = state.Zv[0]
    view_2 = state.Zv[1]

    if view_1 != view_2:
        print("Columns not in same view")
        return

    log_crp = su.get_cluster_crps(state, 0)
    K = len(log_crp)

    X = numpy.zeros(N)
    Y = numpy.zeros(N)

    clusters_col_1 = su.create_cluster_set(state, 0)
    clusters_col_2 = su.create_cluster_set(state, 1)

    for i in range(N):
        c = utils.log_pflip(log_crp)
        x = clusters_col_1[c].predictive_draw()
        y = clusters_col_2[c].predictive_draw()

        X[i] = x
        Y[i] = y

    pylab.scatter(X,Y, color='red', label='inferred')
    pylab.scatter(state.dims[0].X, state.dims[1].X, color='blue', label='actual')
    pylab.show()
