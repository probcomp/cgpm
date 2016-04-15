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

import matplotlib.pyplot as plt
import numpy as np
from math import log

import gpmcc.utils.general as gu
from gpmcc.utils.config import colors

_plot_layout = {
    1: (1,1),
    2: (2,1),
    3: (3,1),
    4: (2,2),
    5: (3,2),
    6: (3,2),
    7: (4,2),
    8: (4,2),
    9: (3,3),
    10: (5,2),
    11: (4,3),
    12: (4,3),
    13: (5,3),
    14: (5,3),
    15: (5,3),
    16: (4,4),
    17: (6,3),
    18: (6,3),
    19: (5,4),
    20: (5,4),
    21: (7,3),
    22: (6,4),
    23: (6,4),
    24: (6,4),
}

def get_state_plot_layout(n_cols):
    layout = dict()
    layout['plots_x'] = _plot_layout[n_cols][0]
    layout['plots_y'] = _plot_layout[n_cols][1]
    layout['plot_inches_x'] = 13/6. * layout['plots_x']
    layout['plot_inches_y'] = 6. * layout['plots_y']
    layout['border_color'] = colors()
    return layout

def plot_dist_continuous(X, clusters, ax=None, Y=None, hist=True):
    # Create a new axis?
    if ax is None:
        _, ax = plt.subplots()
    # Set up x axis.
    x_min = min(X)
    x_max = max(X)
    if Y is None:
        Y = np.linspace(x_min, x_max, 200)
    # Compute weighted pdfs.
    K = len(clusters)
    pdf = np.zeros((K, len(Y)))
    W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
    for k in xrange(K):
        pdf[k,:] = np.exp([W[k] + clusters[k].logpdf(y)
                for y in Y])
        color, alpha = gu.curve_color(k)
        ax.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)
    # Plot the sum of pdfs.
    ax.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
    # Plot the samples.
    if hist:
        nbins = min([len(X), 50])
        ax.hist(X, nbins, normed=True, color='black', alpha=.5,
            edgecolor='none')
    else:
        y_max = ax.get_ylim()[1]
        for x in X:
            ax.vlines(x, 0, y_max/10., linewidth=1)
    # Title.
    ax.set_title(clusters[0].name())
    return ax

def plot_dist_discrete(X, clusters, ax=None, Y=None, hist=True):
    # Create a new axis?
    if ax is None:
        _, ax = plt.subplots()
    # Set up x axis.
    x_max = max(X)
    Y = range(int(x_max)+1)
    X_hist = [np.sum(X==i) / float(len(X)) for i in Y]
    ax.bar(Y, X_hist, color='gray', edgecolor='none')
    # Compute weighted pdfs
    K = len(clusters)
    pdf = np.zeros((K, len(Y)))
    W = [log(clusters[k].N) - log(float(len(X))) for k in xrange(K)]
    for k in xrange(K):
        pdf[k,:] = np.exp([W[k] + clusters[k].logpdf(y)
                for y in Y])
        color, alpha = gu.curve_color(k)
        ax.bar(Y, pdf[k,:], color=color, edgecolor='none', alpha=alpha)
    # Plot the sum of pdfs.
    ax.bar(Y, np.sum(pdf, axis=0), color='none', edgecolor='black',
        linewidth=3)
    ax.set_xlim([0, x_max+1])
    # Title.
    ax.set_title(clusters[0].name())
    return ax

def plot_clustermap(D, xticklabels=None, yticklabels=None):
    import seaborn as sns
    if xticklabels is None: xticklabels = range(D.shape[0])
    if yticklabels is None: yticklabels = range(D.shape[1])
    zmat = sns.clustermap(D, yticklabels=yticklabels, xticklabels=xticklabels,
        linewidths=0.2, cmap='BuGn')
    plt.setp(zmat.ax_heatmap.get_yticklabels(), rotation=0)
    plt.setp(zmat.ax_heatmap.get_xticklabels(), rotation=90)
    return zmat

def plot_samples(X, ax=None):
    if ax is None:
        _, ax = plt.subplots()
        ax.set_ylim([0, 10])
    for x in X:
        ax.vlines(x, 0, 1., linewidth=1)
    return ax
