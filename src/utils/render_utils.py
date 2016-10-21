import matplotlib.pyplot as plt
import numpy as np
import seaborn
import matplotlib.gridspec as gridspec

seaborn.set_style("white")

def viz_data(data, ax=None, row_names=None,
             col_names=None, cmap=None):
    """
    Vizualize heterogeneous data.
    Standardize data across columns.
    Ignore nan values (plotted white)
    """
    if isinstance(data, list): data = np.array(data)
    if ax is None: ax = plt.gca()
    if cmap is None: cmap = "YlGn"

    data_normed = nannormalize(data)

    ax.imshow(
        data_normed, interpolation='None', cmap="YlGn",
        vmin=-0.1, vmax=1.1, aspect='auto')

    ax.set_xlim([-0.5, data_normed.shape[1]-0.5])
    ax.set_ylim([-0.5, data_normed.shape[0]-0.5])
    if row_names is not None:
        ax.set_yticks(range(data_normed.shape[0]))
        ax.set_yticklabels(row_names)
        
    if col_names is not None:
        ax.set_xticks(range(data_normed.shape[1]))
        ax.set_xticklabels(col_names, rotation=90)
    return ax


def viz_view(view, ax=None, row_names=None, col_names=None):
    """ 
    Order rows according to clusters and draw line between clusters.
    Visualize this using imshow with two colors only.
    """
    if ax is None: ax = plt.gca()
    if isinstance(row_names, list): row_names = np.array(row_names)

    # Get data restricted to current view's outputs
    data_dict = get_view_data(view)
    data_arr = np.array(data_dict.values()).T

    # Sort data into clusters
    assignments = np.array(view.Zr().values())
    row_indexes = np.argsort(assignments, kind='mergesort')
    clustered_data = data_arr[row_indexes, :]
    cluster_boundaries = (
        assignments[row_indexes][:-1] != assignments[row_indexes][1:]
    ).nonzero()[0]

    # Sort out row and column names
    if row_names is not None:
        ax.set_yticklabels(row_names[row_indexes])
    output_cols = data_dict.keys()
    if col_names is None:
        col_names = output_cols
    elif len(col_names) == len(view.X.values())-1:
        col_names = col_names[output_cols]

    # Plot clustered data 
    ax = viz_data(clustered_data, ax, row_names, col_names)

    # Plot lines between clusters
    for bd in cluster_boundaries:
        ax.plot(
            [-0.5, clustered_data.shape[1]-0.5], [bd+0.5, bd+0.5],
            color='magenta', linewidth=3)
    return ax

def viz_state(state, row_names=None, col_names=None):
    """ 
    For each each view call viz_view as defined above. 
    Plot each view next to each other.
    """
    # Get data and latents
    data_arr = np.array(state.X.values()).T
    D = data_arr.shape[1]
    views = state.views.keys()
    view_assigns = [state.Zv()[i] for i in range(D)]    

    # Function for getting the subplot widths
    def widths(view_assigns, views):
        answer = []
        view_assigns = np.array(view_assigns)
        for view in views:
            (inds,) = np.nonzero(view_assigns == view)
            answer.append(len(inds))
        return answer

    # Create grid for subplots 
    gs = gridspec.GridSpec(
        1, len(views), width_ratios=widths(view_assigns, views))

    # Plot data for each view
    ax = []
    for (i, view) in enumerate(views):
        ax.append(plt.subplot(gs[i]))
        if i == 0:
            ax[-1].set_ylabel("Row", size='x-large')
    
        (inds,) = np.nonzero(np.array(view_assigns) == view)
        ax[-1] = viz_view(state.views[view], ax[-1], row_names, col_names)
        ax[-1].set_xlabel("Column", size='x-large')
        ax[-1].set_title("View %d" % (view), size='x-large')

    plt.subplots_adjust(top=0.84)
    return ax
 

# # Helpers # #
    
def nanptp(array, axis=0):
    """
    Return peak-to-peak distance of an array ignoring nan values.
    """
    ptp = np.nanmax(array, axis=axis) - np.nanmin(array, axis=0)
    ptp_without_null = [i if i != 0 else 1.0 for i in ptp]
    return ptp_without_null
    
def nannormalize(data):
    """
    Normalize data across the columns ignoring nan values.
    """
    return (data - np.nanmin(data, axis=0)) / nanptp(data, axis=0)

def get_view_data(view):
    """
    Gets the columns of the data for which there is an output variable.
    Returns a dict
    """
    exposed_outputs = view.outputs[1:]
    return {key: val for key, val in view.X.iteritems()
            if key in exposed_outputs}
