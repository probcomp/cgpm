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

import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn


seaborn.set_style('white')


def compute_axis_size_viz_data(data, row_names, col_names):
    if isinstance(data, list):
        data = np.array(data)
    row_height = max(map(len, map(str, row_names)))
    col_width = max(map(len, map(str, col_names)))
    height = data.shape[0] #/ 2. + row_height
    width = data.shape[1] #/ 2. + col_width
    return height, width


def viz_data_raw(
        data, ax=None, row_names=None, col_names=None, cmap=None, title=None):
    if isinstance(data, list):
        data = np.array(data)
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = "YlGn"
    if row_names is None:
        row_names = range(data.shape[0])
    if col_names is None:
        col_names = range(data.shape[1])

    height, width = compute_axis_size_viz_data(data, row_names, col_names)

    data_normed = nannormalize(data)

    ax.matshow(
        data_normed, interpolation='None', cmap=cmap,
        vmin=-0.1, vmax=1.1, aspect='auto')

    ax.set_xlim([-0.5, data_normed.shape[1]-0.5])
    ax.set_ylim([-0.5, data_normed.shape[0]-0.5])

    size = np.sqrt(width**2 + height**2)
    yticklabelsize = size - height
    if row_names is not None:
        ax.set_yticks(range(data_normed.shape[0]))
        ax.set_yticklabels(row_names, ha='right', size='x-large')
        # ax.set_yticklabels(row_names, ha='right')

    xticklabelsize = size - width/3.
    if col_names is not None:
        ax.set_xticks(range(data_normed.shape[1]))
        ax.set_xticklabels(
            col_names, rotation=45, rotation_mode='anchor', ha='left',
            size='x-large')
        # ax.set_xticklabels(
        # col_names, rotation=45, rotation_mode='anchor', ha='left')

    # Hack to set grids off-center
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in ax.get_yticks()][1:], minor='true')
    ax.grid(True, which='minor')

    return ax


def viz_data(
        data, ax=None, row_names=None, col_names=None, cmap=None, title=None,
        savefile=None):
    """Vizualize heterogeneous data.

    Standardizes data across columns. Ignores nan values (plotted white).
    """
    if savefile is None: savefile = "viz_data_foo.png"
    if row_names is None: row_names = range(data.shape[0])
    if col_names is None: col_names = range(data.shape[1])

    ax = viz_data_raw(data, ax, row_names, col_names, cmap, title)
    height, width = compute_axis_size_viz_data(data, row_names, col_names)

    fig = ax.get_figure()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    fig.set_tight_layout(True)
    if savefile:
        fig.savefig(savefile)
    return ax

def viz_view_raw(view, ax=None, row_names=None, col_names=None):
    """Order rows according to clusters and draw line between clusters.

    Visualize this using imshow with two colors only.
    """
    if ax is None:
        ax = plt.gca()
    if isinstance(row_names, list):
        row_names = np.array(row_names)
    if isinstance(col_names, list):
        col_names = np.array(col_names)

    # Get data restricted to current view's outputs.
    data_dict = get_view_data(view)
    data_arr = np.array(data_dict.values()).T

    # Partition data into clusters.
    customers = view.Zr()
    tables = set(view.Zr().values())
    clustered_rows_raw = [
        [c for c in customers if customers[c] == k]
        for k in tables
    ]

    # Within a cluster, sort the rows by their predictive likelihood.
    def retrieve_row_score(r):
        data = {
            d: data_dict[d][r] for d in data_dict
            if not np.isnan(data_dict[d][r])
        }
        return view.logpdf(-1, data, {view.outputs[0]: view.Zr(r)})
    row_scores = [
        [retrieve_row_score(r) for r in cluster]
        for cluster in clustered_rows_raw
    ]
    clustered_rows = [
        [r for s, r in sorted(zip(scores, row))]
        for scores, row in zip(row_scores, clustered_rows_raw)
    ]

    # Order the clusters by number of rows.
    clustered_rows_reordered = sorted(clustered_rows, key=len)

    # Retrieve the dataset based on the ordering.
    clustered_data = np.vstack(
        [data_arr[cluster] for cluster in clustered_rows_reordered])

    # Unravel the clusters one long list and find the cluster boundaries.
    row_indexes = list(itertools.chain.from_iterable(clustered_rows_reordered))
    assignments = np.asarray([customers[r] for r in row_indexes])
    cluster_boundaries = np.nonzero(assignments[:-1] != assignments[1:])[0]

    # Retrieve the logscores of the dimensions for sorting.
    scores = [
        [view.dims[d].clusters[k].logpdf_score() for k in tables]
        for d in data_dict
    ]
    dim_scores = np.sum(scores, axis=1)
    dim_ordering = np.argsort(dim_scores)

    # Reorder the columns in the dataset.
    clustered_data = clustered_data[:,dim_ordering]

    # Determine row and column names.
    if row_names is not None:
        row_names = row_names[row_indexes]

    if col_names is None:
        col_names = dim_ordering
    elif len(col_names) == len(view.X.values()):
        col_names = col_names[dim_ordering]

    # Plot clustered data.
    ax = viz_data_raw(clustered_data, ax, row_names, col_names)

    # Plot lines between clusters
    for bd in cluster_boundaries:
        ax.plot(
            [-0.5, clustered_data.shape[1]-0.5], [bd+0.5, bd+0.5],
            color='magenta', linewidth=3)

    return ax


def viz_view(view, ax=None, row_names=None, col_names=None, savefile=None):
    """Order rows according to clusters and draw line between clusters.

    Visualize this using imshow with two colors only.
    """
    # Get data restricted to current view's outputs
    data_dict = get_view_data(view)
    data_arr = np.array(data_dict.values()).T

    if ax is None: ax = plt.gca()
    if row_names is None: row_names = range(data_arr.shape[0])
    if col_names is None: col_names = range(data_arr.shape[1])
    if savefile is None: savefile = "view_foo.png"
    if isinstance(row_names, list): row_names = np.array(row_names)

    ax = viz_view_raw(view, ax, row_names, col_names)

    height, width = compute_axis_size_viz_data(data_arr, row_names, col_names)

    fig = ax.get_figure()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    fig.set_tight_layout(True)
    if savefile:
        fig.savefig(savefile)
    return ax


def viz_state(state, row_names=None, col_names=None, savefile=None):
    """For each each view call viz_view as defined above.

    Plot each view next to each other.
    """
    data_arr = np.array(state.X.values()).T

    if row_names is None:
        row_names = range(data_arr.shape[0])
    if col_names is None:
        col_names = range(data_arr.shape[1])

    views = state.views.keys()
    views = sorted(views, key=lambda v: len(state.views[v].outputs))[::-1]

    # Get the subplot widths
    view_widths = []
    view_heights = []
    for view in views:
        data_view = np.array(
            get_view_data(state.views[view]).values()).T
        height, width = compute_axis_size_viz_data(
            data_view, row_names, col_names)
        view_widths.append(width)
        view_heights.append(1)

    fig = plt.figure(figsize=(32, 18))

    # Create grid for subplots.
    gs = gridspec.GridSpec(1, len(views), width_ratios=view_widths, wspace=1)

    # Plot data for each view
    ax_list = []
    for (i, v) in enumerate(views):
        ax_list.append(fig.add_subplot(gs[i]))

        ax_list[-1] = viz_view_raw(
            state.views[v], ax_list[-1], row_names, col_names)

    plt.subplots_adjust(top=0.84)
    if savefile:
        fig.savefig(savefile)
        plt.close('all')

    return ax_list


# # Helpers # #

def nanptp(array, axis=0):
    """Return peak-to-peak distance of an array ignoring nan values."""
    ptp = np.nanmax(array, axis=axis) - np.nanmin(array, axis=0)
    ptp_without_null = [i if i != 0 else 1. for i in ptp]
    return ptp_without_null

def nannormalize(data):
    """Normalize data across the columns ignoring nan values."""
    return (data - np.nanmin(data, axis=0)) / nanptp(data, axis=0)


def get_view_data(view):
    """Gets the columns of the data for which there is an output variable.

    Returns a dict.
    """
    exposed_outputs = view.outputs[1:]
    return {key: view.X[key] for key in exposed_outputs}
