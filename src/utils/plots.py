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

from math import log

import matplotlib.pyplot as plt
import numpy as np

from cgpm.utils import general as gu
from cgpm.utils.config import colors


_plot_layout = {1: (1,1), 2: (2,1), 3: (3,1), 4: (2,2), 5: (3,2), 6: (3,2),
    7: (4,2), 8: (4,2), 9: (3,3), 10: (5,2), 11: (4,3), 12: (4,3), 13: (5,3),
    14: (5,3), 15: (5,3), 16: (4,4), 17: (6,3), 18: (6,3),  19: (5,4),
    20: (5,4), 21: (7,3), 22: (6,4), 23: (6,4), 24: (6,4),
    }

def get_state_plot_layout(n_cols):
    layout = dict()
    layout['plots_x'] = _plot_layout[n_cols][0]
    layout['plots_y'] = _plot_layout[n_cols][1]
    layout['plot_inches_x'] = 13/6. * layout['plots_x']
    layout['plot_inches_y'] = 6. * layout['plots_y']
    layout['border_color'] = colors()
    return layout

def plot_dist_continuous(X, output, clusters, ax=None, Y=None, hist=True):
    # Create a new axis?
    if ax is None:
        _, ax = plt.subplots()
    # Set up x axis.
    x_min = min(X)
    x_max = max(X)
    if Y is None:
        Y = np.linspace(x_min, x_max, 200)
    # Compute weighted pdfs.
    pdf = np.zeros((len(clusters), len(Y)))
    W = [log(clusters[k].N) - log(float(len(X))) for k in clusters]
    for i, k in enumerate(clusters):
        pdf[i,:] = np.exp(
            [W[i] + clusters[k].logpdf(None, {output:y}) for y in Y])
        color, alpha = gu.curve_color(i)
        ax.plot(Y, pdf[i,:], color=color, linewidth=5, alpha=alpha)
    # Plot the sum of pdfs.
    ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
    # Plot the samples.
    if hist:
        nbins = min([len(X), 50])
        ax.hist(
            X, nbins, normed=True, color='black', alpha=.5, edgecolor='none')
    else:
        y_max = ax.get_ylim()[1]
        for x in X:
            ax.vlines(x, 0, y_max/10., linewidth=1)
    # Title.
    ax.set_title(clusters.values()[0].name())
    return ax

def plot_dist_discrete(X, output, clusters, ax=None, Y=None, hist=True):
    # Create a new axis?
    if ax is None:
        _, ax = plt.subplots()
    # Set up x axis.
    X = np.asarray(X, dtype=int)
    x_max = max(X)
    Y = range(int(x_max)+1)
    X_hist = np.bincount(X) / float(len(X))
    ax.bar(Y, X_hist, color='gray', edgecolor='none')
    # Compute weighted pdfs
    pdf = np.zeros((len(clusters), len(Y)))
    W = [log(clusters[k].N) - log(float(len(X))) for k in clusters]
    for i, k in enumerate(clusters):
        pdf[i,:] = np.exp(
            [W[i] + clusters[k].logpdf(None, {output:y}) for y in Y])
        color, alpha = gu.curve_color(i)
        ax.bar(Y, pdf[i,:], color=color, edgecolor='none', alpha=alpha)
    # Plot the sum of pdfs.
    ax.bar(
        Y, np.sum(pdf, axis=0), color='none', edgecolor='black', linewidth=3)
    ax.set_xlim([0, x_max+1])
    # Title.
    ax.set_title(clusters.values()[0].name())
    return ax

def plot_clustermap(D, xticklabels=None, yticklabels=None):
    import seaborn as sns
    if xticklabels is None: xticklabels = range(D.shape[0])
    if yticklabels is None: yticklabels = range(D.shape[1])
    zmat = sns.clustermap(
        D, yticklabels=yticklabels, xticklabels=xticklabels,
        linewidths=0.2, cmap='BuGn')
    plt.setp(zmat.ax_heatmap.get_yticklabels(), rotation=0)
    plt.setp(zmat.ax_heatmap.get_xticklabels(), rotation=90)
    return zmat

def clustermap_ordering(D):
    zmat = plot_clustermap(D)
    plt.close(zmat.fig)
    return zmat.dendrogram_row.reordered_ind

def plot_heatmap(
        D, xordering=None, yordering=None, xticklabels=None,
        yticklabels=None, vmin=None, vmax=None, ax=None):
    import seaborn as sns
    D = np.copy(D)

    if ax is None:
        _, ax = plt.subplots()
    if xticklabels is None:
        xticklabels = np.arange(D.shape[0])
    if yticklabels is None:
        yticklabels = np.arange(D.shape[1])
    if xordering is not None:
        xticklabels = xticklabels[xordering]
        D = D[:,xordering]
    if yordering is not None:
        yticklabels = yticklabels[yordering]
        D = D[yordering,:]

    sns.heatmap(
        D, yticklabels=yticklabels, xticklabels=xticklabels,
        linewidths=0.2, cmap='BuGn', ax=ax, vmin=vmin, vmax=vmax)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(yticklabels, rotation=0)
    return ax

def plot_samples(X, ax=None):
    if ax is None:
        _, ax = plt.subplots()
        ax.set_ylim([0, 10])
    for x in X:
        ax.vlines(x, 0, 1., linewidth=1)
    return ax

def partition_to_zmatrix(Zv, ordering=None):
    """Convert a cgpm.crosscat.State view partition Zv into a binary zmatrix."""
    # Default ordering of columns by increasing index.
    if ordering is None:
        ordering = sorted(Zv)

    # Converts a column index to its 0-based index in ordering.
    column_to_index = {col: ordering.index(col) for col in Zv}

    # block_vectors[i] is a binary vector, with 1 for columns in that view.
    views = set(Zv.values())
    block_vectors = {view: np.zeros(len(Zv)) for view in views}
    for view in views:
        cols = [column_to_index[c] for c, v in Zv.iteritems() if v == view]
        block_vectors[view][cols] = 1

    D = np.zeros((len(Zv), len(Zv)))
    for col in Zv:
        D[column_to_index[col]] = block_vectors[Zv[col]]

    return D

def partitions_to_zmatrix(Zvs, ordering=None):
    """Converts a collection cgpm.crosscat.State view partitions Zvs
    into a real-valued zmatrix, which is the mean of all Zv."""
    Ds = [partition_to_zmatrix(Zv, ordering=ordering) for Zv in Zvs]
    return np.mean(Ds, axis=0)


def plot_logscore(logscores, ax=None):
    assert all(len(l) == len(logscores[0]) for l in logscores)
    if ax is None:
        fig, ax = plt.subplots()

    for logscore in logscores:
        ax.plot(range(len(logscores[0])), logscore)

    ax.set_xlabel('Number of Full Gibbs Sweeps')
    ax.set_ylabel('Log Score')
    ax.grid()
    return ax


def engine_to_zmatrix_history(engine, ordering=None):
    num_transitions = len(engine.states[0].diagnostics['column_partition'])
    Zvs = [[dict(state.diagnostics['column_partition'][i])
        for state in engine.states] for i in xrange(num_transitions)]

    # Find the ordering at the final step.
    if ordering is None:
        D = partitions_to_zmatrix(Zvs[-1])
        ordering = clustermap_ordering(D)

    # Return the history of zmatrices.
    return [partitions_to_zmatrix(Z, ordering=ordering) for Z in Zvs]
