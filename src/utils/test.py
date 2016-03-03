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

import numpy as np
import math

from scipy.stats import norm
from scipy.stats import geom

from gpmcc import dim
import gpmcc.utils.general as gu
import gpmcc.utils.config as cu

def gen_data_table(n_rows, view_weights, cluster_weights, cctypes, distargs,
        separation):
    """Generates data, partitions, and Dim.

     Parameters
     ----------
     n_rows : int
        Mumber of rows (data points) to generate.
     view_weights : np.ndarray
        An n_views length list of floats that sum to one. The weights indicate
        the proportion of columns in each view.
    cluster_weights : np.ndarray
        An n_views length list of n_cluster length lists that sum to one.
        The weights indicate the proportion of rows in each cluster.
     cctypes : list<str>
        n_columns length list of string specifying the distribution types for
        each column.
     distargs : list
        List of distargs for each column (see documentation for each data type
            for info on distargs).
     separation : list
        An n_cols length list of values between [0,1], where seperation[i] is
        the seperation of clusters in column i. Values closer to 1 imply higher
        seperation.

     Returns
     -------
     T : np.ndarray
        An (n_cols, n_rows) matrix, where each row T[i,:] is the data for
        column i (tranpose of a design matrix).
    Zv : list
        An n_cols length list of integers, where Zv[i] is the view assignment
        of column i.
    Zc : list<list>
        An n_view length list of lists, where Zc[v][r] is the cluster assignment
        of row r in view v.

    Example
    -------
    >>> n_rows = 500
    >>> view_weights = [.2, .8]
    >>> cluster_weights = [[.3, .2, .5], [.4, .6]]
    >>> cctypes = ['lognormal','normal','poisson','categorical',
    ...     'vonmises', 'bernoulli']
    >>> distargs = [None, None, None, {'k':8}, None, None]
    >>> separation = [.8, .7, .9, .6, .7, .85]
    >>> T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights,
    ...     cluster_weights, dists, distargs, separation)
    """
    n_cols = len(cctypes)
    Zv, Zc = gen_partition_from_weights(n_rows, n_cols, view_weights,
        cluster_weights)
    T = np.zeros((n_cols, n_rows))

    for col in xrange(n_cols):
        cctype = cctypes[col]
        args = distargs[col]
        view = Zv[col]
        Tc = _gen_data[cctype](Zc[view], separation[col], distargs=args)
        T[col] = Tc

    return T, Zv, Zc

def gen_dims_from_structure(T, Zv, Zc, cctypes, distargs):
    n_cols = len(Zv)
    dims = []
    for col in xrange(n_cols):
        v = Zv[col]
        cctype = cctypes[col]
        dim_c = dim.Dim(cctype, col, distargs=distargs[col])
        dim_c.transition_hyper_grids(T[col])
        dim_c.bulk_incorporate(T[col], Zc[v])
        dims.append(dim_c)
    return dims

def _gen_beta_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)

    K = np.max(Z)+1
    alphas = np.linspace(.5 -.5*separation*.85, .5 + .5*separation*.85, K)
    Tc = np.zeros(n_rows)

    for r in xrange(n_rows):
        cluster = Z[r]
        alpha = alphas[cluster]
        beta = (1.0-alpha)*20.0*(norm.pdf(alpha,.5,.25))
        alpha *= 20.0*norm.pdf(alpha,.5,.25)
        # beta *= 10.0
        Tc[r] = np.random.beta(alpha, beta)

    return Tc

def _gen_normal_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)

    Tc = np.zeros(n_rows)
    for r in xrange(n_rows):
        cluster = Z[r]
        mu = cluster*(5.0*separation)
        sigma = 1.0
        Tc[r] = np.random.normal(loc=mu, scale=sigma)

    return Tc

def _gen_normal_trunc_data(Z, separation=.9, distargs=None):
    l, h = distargs['l'], distargs['h']
    max_draws = 100
    n_rows = len(Z)

    K = max(Z) + 1
    mean = l+h/2.

    bins = np.linspace(l, h, K+1)
    bin_centers = [.5*(bins[i-1]+bins[i]) for i in xrange(1, len(bins))]
    distances = [mean - bc for bc in bin_centers]
    mus = [bc+(1-separation)*d for bc, d in zip(bin_centers, distances)]

    Tc = np.zeros(n_rows)
    for r in xrange(n_rows):
        cluster = Z[r]
        sigma = 1
        i = 0
        while True:
            i += 1
            x = np.random.normal(loc=mus[cluster], scale=sigma)
            if l<=x<=h:
                break
            if i > max_draws:
                raise ValueError('Could not generate normal_trunc data.')
        Tc[r] = x

    return Tc

def _gen_vonmises_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)

    num_clusters =  max(Z)+1
    sep = (2*math.pi/num_clusters)

    mus = [c*sep for c in xrange(num_clusters)]
    std = sep/(5.0*separation**.75)
    k = 1/(std*std)

    Tc = np.zeros(n_rows)
    for r in xrange(n_rows):
        cluster = Z[r]
        mu = mus[cluster]
        Tc[r] = np.random.vonmises(mu, k) + math.pi

    return Tc

def _gen_poisson_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)

    for r in xrange(n_rows):
        cluster = Z[r]
        lam = (cluster)*(4.0*separation)+1
        Tc[r] = np.random.poisson(lam)

    return Tc

def _gen_exponential_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)

    for r in xrange(n_rows):
        cluster = Z[r]
        mu = (cluster)*(4.0*separation)+1
        Tc[r] = np.random.exponential(mu)

    return Tc

def _gen_geometric_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)
    K = np.max(Z)+1

    ps = np.linspace(.5 -.5*separation*.85, .5 + .5*separation*.85, K)
    Tc = np.zeros(n_rows)
    for r in xrange(n_rows):
        cluster = Z[r]
        Tc[r] = geom.rvs(ps[cluster], loc=-1)

    return Tc

def _gen_lognormal_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)

    if separation > .9:
        separation = .9

    Tc = np.zeros(n_rows)
    for r in xrange(n_rows):
        cluster = Z[r]
        mu = cluster*(.9*separation**2)
        Tc[r] = np.random.lognormal(mean=mu,
            sigma=(1.0-separation)/(cluster+1.0))

    return Tc

def _gen_bernoulli_data(Z, separation=.9, distargs=None):
    n_rows = len(Z)

    Tc = np.zeros(n_rows)
    K = max(Z)+1
    thetas = np.linspace(0.0,separation,K)

    for r in range(n_rows):
        cluster = Z[r]
        theta = thetas[cluster]
        x = 0.0
        if np.random.random() < theta:
            x = 1.0
        Tc[r] = x

    return Tc

def _gen_categorical_data(Z, separation=.9, distargs=None):
    k = distargs['k']
    n_rows = len(Z)

    if separation > .95:
        separation = .95

    Tc = np.zeros(n_rows, dtype=int)
    C = max(Z)+1
    theta_arrays = [np.random.dirichlet(np.ones(k)*(1.0-separation), 1)
        for _ in range(C)]

    for r in xrange(n_rows):
        cluster = Z[r]
        thetas = theta_arrays[cluster][0]
        x = int(gu.pflip(thetas))
        Tc[r] = x
    return Tc

def gen_partition_from_weights(n_rows, n_cols, view_weights, clusters_weights):
    n_views = len(view_weights)
    Zv = [v for v in range(n_views)]
    for _ in xrange(n_cols - n_views):
        v = gu.pflip(view_weights)
        Zv.append(v)

    np.random.shuffle(Zv)
    assert len(Zv) == n_cols

    Zc = []
    for v in xrange(n_views):
        n_clusters = len(clusters_weights[v])
        Z = [c for c in xrange(n_clusters)]
        for _ in range(n_rows-n_clusters):
            c_weights = np.copy(clusters_weights[v])
            c = gu.pflip(c_weights)
            Z.append(c)
        np.random.shuffle(Z)
        Zc.append(Z)

    assert len(Zc) == n_views
    assert len(Zc[0]) == n_rows

    return Zv, Zc

def gen_partition_crp(n_rows, n_cols, n_views, alphas):
    Zv = [v for v in range(n_views)]
    for _ in range(n_cols-n_views):
        Zv.append(np.random.randrange(n_views))
    np.random.shuffle(Zv)
    Zc = []
    for v in range(n_views):
        Zc.append(gu.simulate_crp(n_rows, alphas[v]))

    return Zv, Zc

def column_average_ari(Zv, Zc, cc_state_object):
    from sklearn.metrics import adjusted_rand_score
    ari = 0
    n_cols = len(Zv)
    for col in xrange(n_cols):
        view_t = Zv[col]
        Zc_true = Zc[view_t]

        view_i = cc_state_object.Zv[col]
        Zc_inferred = cc_state_object.views[view_i].Z.tolist()
        ari += adjusted_rand_score(Zc_true, Zc_inferred)

    return ari/float(n_cols)

def gen_linear(N, noise=.1, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    return rng.multivariate_normal([0,0], cov=[[1,1-noise],[1-noise,1]], size=N)

def gen_x(N, noise=.1, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    X = np.zeros((N,2))
    for i in xrange(N):
        if rng.rand() < .5:
            cov = np.array([[1,1-noise],[1-noise,1]])
        else:
            cov = np.array([[1,-1+noise],[-1+noise,1]])
        x = rng.multivariate_normal([0,0], cov=cov)
        X[i,:] = x
    return X

def gen_sin(N, noise=.1, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    x_range = [-3.*math.pi/2., 3.*math.pi/2.]
    X = np.zeros((N,2))
    for i in xrange(N):
        x = rng.uniform(x_range[0], x_range[1])
        y = math.cos(x) + rng.uniform(-noise, noise)
        X[i,0] = x
        X[i,1] = y
    return X

def gen_ring(N, noise=.1, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    X = np.zeros((N,2))
    for i in xrange(N):
        angle = rng.uniform(0.0, 2.0*math.pi)
        distance = rng.uniform(1.0-noise, 1.0)
        X[i,0] = math.cos(angle)*distance
        X[i,1] = math.sin(angle)*distance
    return X

def gen_dots(N=200, noise=.1, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    X = np.zeros((N,2))
    mx = [ -1, 1, -1, 1]
    my = [ -1, -1, 1, 1]
    for i in range(N):
        n = rng.randint(4)
        x = rng.normal(loc=mx[n], scale=noise) if noise > 0 else mx[n]
        y = rng.normal(loc=my[n], scale=noise) if noise > 0 else my[n]
        X[i,0] = x
        X[i,1] = y
    return X

_gen_data = {
    'bernoulli'         : _gen_bernoulli_data,
    'beta_uc'           : _gen_beta_data,
    'categorical'       : _gen_categorical_data,
    'exponential'       : _gen_exponential_data,
    'geometric'         : _gen_geometric_data,
    'lognormal'         : _gen_lognormal_data,
    'normal'            : _gen_normal_data,
    'normal_trunc'      : _gen_normal_trunc_data,
    'poisson'           : _gen_poisson_data,
    'vonmises'          : _gen_vonmises_data,
}
