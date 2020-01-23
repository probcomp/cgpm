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

from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import math

import numpy as np

from scipy.stats import norm

from cgpm.crosscat.engine import Engine
from cgpm.crosscat.state import State
from cgpm.mixtures.dim import Dim
from cgpm.mixtures.view import View
from cgpm.utils import general as gu


def gen_data_table(n_rows, view_weights, cluster_weights, cctypes, distargs,
        separation, view_partition=None, rng=None):
    """Generates data, partitions, and Dim.

     Parameters
     ----------
     n_rows : int
        Mumber of rows (data points) to generate.
     view_weights : list<float>
        An n_views length list of floats that sum to one. The weights indicate
        the proportion of columns in each view.
    cluster_weights : list<list<float>>
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
    if rng is None:
        rng = gu.gen_rng()

    n_cols = len(cctypes)

    if view_partition:
        Zv = list(view_partition)
    else:
        Zv = gen_partition(n_cols, view_weights, rng)

    Zc = [gen_partition(n_rows, cw, rng) for cw in cluster_weights]

    assert len(Zv) == n_cols
    assert len(Zc) == len(set(Zv))
    assert len(Zc[0]) == n_rows

    T = np.zeros((n_cols, n_rows))

    for col in range(n_cols):
        cctype = cctypes[col]
        args = distargs[col]
        view = Zv[col]
        Tc = _gen_data[cctype](
            Zc[view], rng, separation=separation[col], distargs=args)
        T[col] = Tc

    return T, Zv, Zc

def gen_dims_from_structure(T, Zv, Zc, cctypes, distargs):
    n_cols = len(Zv)
    dims = []
    for col in range(n_cols):
        v = Zv[col]
        cctype = cctypes[col]
        dim_c = Dim(cctype, col, distargs=distargs[col])
        dim_c.transition_hyper_grids(T[col])
        dim_c.bulk_incorporate(T[col], Zc[v])
        dims.append(dim_c)
    return dims

def _gen_beta_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    K = np.max(Z)+1
    alphas = np.linspace(.5 - .5*separation*.85, .5 + .5*separation*.85, K)
    Tc = np.zeros(n_rows)

    for r in range(n_rows):
        cluster = Z[r]
        alpha = alphas[cluster]
        beta = (1.-alpha) * 20.* (norm.pdf(alpha, .5, .25))
        alpha *= 20. * norm.pdf(alpha, .5, .25)
        Tc[r] = rng.beta(alpha, beta)

    return Tc

def _gen_normal_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        mu = cluster * (5.*separation)
        sigma = 1.0
        Tc[r] = rng.normal(loc=mu, scale=sigma)

    return Tc

def _gen_normal_trunc_data(Z, rng, separation=.9, distargs=None):
    l, h = distargs['l'], distargs['h']
    max_draws = 100
    n_rows = len(Z)

    K = max(Z) + 1
    mean = (l+h)/2.

    bins = np.linspace(l, h, K+1)
    bin_centers = [.5*(bins[i-1]+bins[i]) for i in range(1, len(bins))]
    distances = [mean - bc for bc in bin_centers]
    mus = [bc + (1-separation)*d for bc, d in zip(bin_centers, distances)]

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        sigma = 1
        i = 0
        while True:
            i += 1
            x = rng.normal(loc=mus[cluster], scale=sigma)
            if l <= x <= h:
                break
            if max_draws < i:
                raise ValueError('Could not generate normal_trunc data.')
        Tc[r] = x

    return Tc

def _gen_vonmises_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    num_clusters = max(Z)+1
    sep = old_div(2*math.pi, num_clusters)

    mus = [c*sep for c in range(num_clusters)]
    std = old_div(sep,(5.*separation**.75))
    k = old_div(1, (std*std))

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        mu = mus[cluster]
        Tc[r] = rng.vonmises(mu, k) + math.pi

    return Tc

def _gen_poisson_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)

    for r in range(n_rows):
        cluster = Z[r]
        lam = cluster * (4.*separation) + 1
        Tc[r] = rng.poisson(lam)

    return Tc

def _gen_exponential_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)

    for r in range(n_rows):
        cluster = Z[r]
        mu = cluster * (4.*separation) + 1
        Tc[r] = rng.exponential(mu)

    return Tc

def _gen_geometric_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)
    K = np.max(Z)+1

    ps = np.linspace(.5 - .5*separation*.85, .5 + .5*separation*.85, K)
    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        Tc[r] = rng.geometric(ps[cluster]) -1

    return Tc

def _gen_lognormal_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    if separation > .9:
        separation = .9

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        mu = cluster * (.9*separation**2)
        Tc[r] = rng.lognormal(mean=mu, sigma=old_div((1.-separation),(cluster+1.)))

    return Tc

def _gen_bernoulli_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    Tc = np.zeros(n_rows)
    K = max(Z)+1
    thetas = np.linspace(0., separation, K)

    for r in range(n_rows):
        cluster = Z[r]
        theta = thetas[cluster]
        x = 0.0
        if rng.rand() < theta:
            x = 1.0
        Tc[r] = x

    return Tc

def _gen_categorical_data(Z, rng, separation=.9, distargs=None):
    k = int(distargs['k'])
    n_rows = len(Z)

    if separation > .95:
        separation = .95

    Tc = np.zeros(n_rows, dtype=int)
    C = max(Z)+1
    theta_arrays = [rng.dirichlet(np.ones(k)*(1.-separation), 1)
        for _ in range(C)]

    for r in range(n_rows):
        cluster = Z[r]
        thetas = theta_arrays[cluster][0]
        x = gu.pflip(thetas, rng=rng)
        Tc[r] = int(x)
    return Tc

def gen_partition(N, weights, rng):
    assert all(w != 0 for w in weights)
    assert np.allclose(sum(weights), 1)
    K = len(weights)
    assert K <= N    # XXX FIXME
    Z = list(range(K))
    Z.extend(int(gu.pflip(weights, rng=rng)) for _ in range(N-K))
    rng.shuffle(Z)
    return Z

def column_average_ari(Zv, Zc, cc_state_object):
    from sklearn.metrics import adjusted_rand_score
    ari = 0
    n_cols = len(Zv)
    for col in range(n_cols):
        view_t = Zv[col]
        Zc_true = Zc[view_t]

        view_i = cc_state_object.Zv[col]
        Zc_inferred = cc_state_object.views[view_i].Z.tolist()
        ari += adjusted_rand_score(Zc_true, Zc_inferred)

    return ari/float(n_cols)

def gen_simple_engine(multiprocess=1):
    data = np.array([[1, 1, 1]])
    R = len(data)
    D = len(data[0])
    outputs = list(range(D))
    engine = Engine(
        X=data,
        num_states=20,
        rng=gu.gen_rng(1),
        multiprocess=multiprocess,
        outputs=outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        distargs={i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zv={0: 0, 1: 0, 2: 1},
        view_alphas=[1.]*D,
        Zrv={0: [0]*R, 1: [0]*R})
    return engine

def gen_simple_state():
    data = np.array([[1, 1, 1]])
    R = len(data)
    D = len(data[0])
    outputs = list(range(D))
    state = State(
        X=data,
        outputs=outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        hypers=[{'alpha': 1., 'beta': 1.} for i in outputs],
        Zv={0: 0, 1: 0, 2: 1},
        view_alphas=[1.]*D,
        Zrv={0: [0]*R, 1: [0]*R})
    return state

def gen_simple_view():
    data = np.array([[1, 1]])
    R = len(data)
    D = len(data[0])
    outputs = list(range(D))
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    Zr = [0 for i in range(R)]
    view = View(
        X,
        outputs=[1000] + outputs,
        alpha=1.,
        cctypes=['bernoulli']*D,
        hypers={i: {'alpha': 1., 'beta': 1.} for i in outputs},
        Zr=Zr)
    return view

def gen_multitype_view():
    data = np.array([[0, 2, 3.14],
                     [1, 1, 1]])
    D = len(data[0])
    dpm_outputs = list(range(D))
    X = {c: data[:, i].tolist() for i, c in enumerate(dpm_outputs)}
    crp_alpha = 1.
    cctypes = ['bernoulli', 'categorical', 'normal']
    hypers = {
        0: {'alpha': 1., 'beta': 1.},
        1: {'alpha': 1.},
        2: {'m': 0, 'r': 1., 's': 1., 'nu': 1.}}
    distargs = {0: {}, 1: {'k': 3}, 2: {}}
    Zr = [0, 1]
    view = View(
        X,
        outputs=[1000] + dpm_outputs,
        alpha=crp_alpha,
        cctypes=cctypes,
        hypers=hypers,
        distargs=distargs,
        Zr=Zr)
    return view

def change_column_hyperparameters(cgpm, value):
    """Generate new cgpm from input cgpm, and alter its distribution
    hyperparameters to extreme values.
    """
    # Retrieve metadata from cgpm
    metadata = cgpm.to_metadata()

    # Alter metadata.hypers to extreme values according to data type
    columns = list(range(len(metadata['outputs'])))
    if isinstance(cgpm, View):
        columns.pop(-1)  # first output is exposed latent

    cctypes = metadata['cctypes']

    new_hypers = [{} for c in columns]
    for c in columns:
        if cctypes[c] == 'bernoulli':
            new_hypers[c] = {'alpha': value, 'beta': value}

        elif cctypes[c] == 'categorical':
            new_hypers[c] = {'alpha': value}

        elif cctypes[c] == 'normal':
            new_hypers[c] = {'m': value, 'nu': value, 'r': value, 's': value}

        else:
            ValueError('''
               cctype not recognized (not bernoulli, categorical or normal);
            ''')

    # Create a new cgpm from the altered metadata
    new_metadata = metadata
    metadata['hypers'] = new_hypers
    return cgpm.from_metadata(new_metadata)

def change_concentration_hyperparameters(cgpm, value):
    """Generate new cgpm from input cgpm, and alter its crp
    concentration hyperparameter to extreme values.
    """
    # Retrieve metadata from cgpm
    metadata = cgpm.to_metadata()

    # Alter metadata.alpha to extreme values according to data type
    new_metadata = metadata
    if isinstance(cgpm, View):
        new_metadata['alpha'] = value
    elif isinstance(cgpm, State):
        old_view_alphas = metadata['view_alphas']
        new_metadata['view_alphas'] = [(a[0], value) for a in old_view_alphas]
    return cgpm.from_metadata(new_metadata)

def restrict_evidence_to_query(query, evidence):
    """Return subset of evidence whose rows are also present in query."""
    return {i: j for i, j in evidence.items() if i in list(query.keys())}

_gen_data = {
    'bernoulli'         : _gen_bernoulli_data,
    'beta'              : _gen_beta_data,
    'categorical'       : _gen_categorical_data,
    'exponential'       : _gen_exponential_data,
    'geometric'         : _gen_geometric_data,
    'lognormal'         : _gen_lognormal_data,
    'normal'            : _gen_normal_data,
    'normal_trunc'      : _gen_normal_trunc_data,
    'poisson'           : _gen_poisson_data,
    'vonmises'          : _gen_vonmises_data,
}
