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
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import seaborn
seaborn.set_style('white')

from cgpm.utils import timer as tu
from cgpm.utils import general as gu


def get_fig_axes(ax=None, nrows=1, ncols=1):
    if ax is None:
        return plt.subplots(nrows=nrows, ncols=ncols)
    return ax.get_figure(), ax


def compute_axis_size_viz_data(data, **kwargs):
    height = kwargs.get('subsample', data.shape[0]) #/ 2. + row_height
    width = data.shape[1] #/ 2. + col_width
    return height, width


def viz_data_raw(data, ax=None, row_names=None, col_names=None, **kwargs):
    fig, ax = get_fig_axes(ax)
    if isinstance(data, list):
        data = np.array(data)
    if row_names is None:
        row_names = map(str, range(data.shape[0]))
    if col_names is None:
        col_names = map(str, range(data.shape[1]))

    height, width = compute_axis_size_viz_data(data)

    data_normed = nannormalize(data)

    ax.matshow(
        data_normed,
        interpolation='None',
        cmap=kwargs.get('cmap', 'YlGn'),
        vmin=-0.1,
        vmax=1.1,
        aspect='auto'
    )

    ax.set_xlim([-0.5, data_normed.shape[1]-0.5])
    ax.set_ylim([-0.5, data_normed.shape[0]-0.5])

    size = np.sqrt(width**2 + height**2)
    if row_names is not None:
        ax.set_yticks(range(data_normed.shape[0]))
        ax.set_yticklabels(
            row_names,
            ha='right',
            size=kwargs.get('yticklabelsize', size - height),
            rotation_mode='anchor'
        )

    if col_names is not None:
        ax.set_xticks(range(data_normed.shape[1]))
        ax.set_xticklabels(
            col_names,
            rotation=45,
            rotation_mode='anchor',
            ha='left',
            size=kwargs.get('xticklabelsize', 'x-large'),
            fontweight='bold'
        )

    # Hack to set grids off-center.
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in ax.get_yticks()][1:], minor='true')
    ax.grid(True, which='minor')

    return fig, ax


def viz_data(data, ax=None, row_names=None, col_names=None, **kwargs):
    """Visualize heterogeneous data.

    Standardizes data across columns. Ignores nan values (plotted white).
    """
    if row_names is None:
        row_names = map(str, range(data.shape[0]))
    if col_names is None:
        col_names = map(str, range(data.shape[1]))

    fig, ax = viz_data_raw(np.array(data), ax, row_names, col_names, **kwargs)

    height, width = compute_axis_size_viz_data(data, **kwargs)
    fig.set_size_inches((width, height))
    fig.set_tight_layout(True)

    return fig, ax

def viz_view_raw(view, ax=None, row_names=None, col_names=None, **kwargs):
    """Order rows according to clusters and draw line between clusters.

    Visualize using imshow with two colors only.
    """
    if isinstance(row_names, list):
        row_names = np.array(map(str, row_names))
    if isinstance(col_names, list):
        col_names = np.array(map(str, col_names))

    # Retrieve the dataset restricted to this view's outputs.
    data_dict = get_view_data(view)
    data_arr = np.array(data_dict.values()).T


    # Construct rowid -> table mapping.
    crp_lookup = view.Zr()

    # Determine whether sub-sampling and react.
    subsample = int(kwargs.get('subsample', 0))
    seed = int(kwargs.get('seed', 1))
    if subsample and subsample < len(data_arr):
        rng = gu.gen_rng(seed)
        rowids_subsample = rng.choice(
            range(len(data_arr)),
            replace=False,
            size=subsample,
        )
        crp_lookup = {rowid : crp_lookup[rowid] for rowid in rowids_subsample}

    # Unique CRP tables in the subsample.
    crp_tables = set(crp_lookup.values())

    # Unique CRP tables across all rows (for sorting).
    crp_tables_all = set(view.Zr().values())

    # Partition rows by cluster assignment.
    clustered_rows_raw = [
        [rowid for rowid in crp_lookup if crp_lookup[rowid] == table]
        for table in crp_tables
    ]

    # Within a cluster, sort the rows by their predictive likelihood.
    def retrieve_row_score(rowid):
        data = {
            dim: data_dict[dim][rowid] for dim in data_dict
            if not np.isnan(data_dict[dim][rowid])
        }
        return view.logpdf(-1, data, {view.outputs[0]: view.Zr(rowid)})
    row_scores = [
        [retrieve_row_score(rowid) for rowid in cluster]
        for cluster in clustered_rows_raw
    ]
    clustered_rows = [
        [rowid for _score, rowid in sorted(zip(scores, rowids))]
        for scores, rowids in zip(row_scores, clustered_rows_raw)
    ]

    # Order the clusters by number of rows.
    clustered_rows_reordered = sorted(clustered_rows, key=len)

    # Retrieve the dataset based on the ordering.
    clustered_data = np.vstack(
        [data_arr[cluster] for cluster in clustered_rows_reordered])

    # Unravel the clusters one long list.
    row_indexes = list(itertools.chain.from_iterable(clustered_rows_reordered))
    assignments = np.asarray([crp_lookup[rowid] for rowid in row_indexes])

    # Find the cluster boundaries.
    cluster_boundaries = np.nonzero(assignments[:-1] != assignments[1:])[0]

    # Retrieve the logscores of the dimensions for sorting. Note the iteration
    # is over all the tables in the CRP, not those only in the subsample, for
    # consistent ordering irrespective of the random subsample.
    get_dim_score = lambda dim: [
        view.dims[dim].clusters[table].logpdf_score()
        for table in crp_tables_all
    ]
    scores = map(get_dim_score, data_dict.iterkeys())
    dim_scores = np.sum(scores, axis=1)
    dim_ordering = np.argsort(dim_scores)

    # Reorder the columns in the dataset.
    clustered_data = clustered_data[:,dim_ordering]

    # Determine row and column names.
    if row_names is not None:
        row_names = row_names[row_indexes]

    if col_names is None:
        col_names = view.outputs[1:]
    col_names = col_names[dim_ordering]

    # Plot clustered data.
    fig, ax = viz_data_raw(clustered_data, ax, row_names, col_names, **kwargs)

    # Plot lines between clusters
    for bd in cluster_boundaries:
        ax.plot(
            [-0.5, clustered_data.shape[1]-0.5],
            [bd+0.5, bd+0.5],
            color='magenta',
            linewidth=3,
        )

    return fig, ax


def viz_view(view, ax=None, row_names=None, col_names=None, **kwargs):
    """Order rows according to clusters and draw line between clusters.

    Visualize this using imshow with two colors only.
    """
    # Get data restricted to current view's outputs
    data_dict = get_view_data(view)
    data_arr = np.array(data_dict.values()).T

    if row_names is None:
        row_names = map(str, range(data_arr.shape[0]))
    if col_names is None:
        col_names = map(str, range(data_arr.shape[1]))

    if isinstance(row_names, list):
        row_names = np.array(row_names)

    fig, ax = viz_view_raw(view, ax, row_names, col_names, **kwargs)

    height, width = compute_axis_size_viz_data(data_arr, **kwargs)
    fig.set_size_inches((width, height))
    fig.set_tight_layout(True)

    return fig, ax


def viz_state(state, row_names=None, col_names=None, progress=None, **kwargs):
    """Calls viz_view on each view in the state.

    Plot views next to one another.
    """
    data_arr = np.array(state.X.values()).T

    if row_names is None:
        row_names = map(str, range(data_arr.shape[0]))
    if col_names is None:
        col_names = map(str, range(data_arr.shape[1]))

    fig = plt.figure()
    fig.set_size_inches(32, 18)

    views = state.views.keys()
    views = sorted(views, key=lambda v: len(state.views[v].outputs))[::-1]

    # Retrieve the subplot widths.
    view_widths = []
    view_heights = []
    for view in views:
        data_view = np.array(get_view_data(state.views[view]).values()).T
        _height, width = compute_axis_size_viz_data(data_view, **kwargs)
        view_widths.append(width)
        view_heights.append(1)

    # Create grid for subplots and axes.
    gs = gridspec.GridSpec(1, len(views), width_ratios=view_widths, wspace=1)
    axes = [fig.add_subplot(gs[i]) for i in xrange(len(views))]

    # Plot data for each view
    row_names_trim = [row_name[:8] for row_name in row_names]
    for i, (ax, v) in enumerate(zip(axes, views)):
        # Find the columns applicable to this view.
        col_names_v = [
            col_names[state.outputs.index(o)]
            for o in state.views[v].outputs[1:]
        ]
        row_names_v = row_names if i == 0 else row_names_trim
        viz_view_raw(state.views[v], ax, row_names_v, col_names_v, **kwargs)
        if progress:
            tu.progress((float(i)+1)/len(views), sys.stdout)
    if progress:
        sys.stdout.write('\n')

    plt.subplots_adjust(top=0.84)
    return fig, axes


# # Helpers # #

def nanptp(array, axis=0):
    """Returns peak-to-peak distance of an array ignoring nan values."""
    ptp = np.nanmax(array, axis=axis) - np.nanmin(array, axis=0)
    ptp_without_null = [i if i != 0 else 1. for i in ptp]
    return ptp_without_null

def nannormalize(data):
    """Normalizes data across the columns, ignoring nan values."""
    return (data - np.nanmin(data, axis=0)) / nanptp(data, axis=0)


def get_view_data(view):
    """Returns the columns of the data for which there is an output variable.

    Returns a dict.
    """
    exposed_outputs = view.outputs[1:]
    return {key: view.X[key] for key in exposed_outputs}
