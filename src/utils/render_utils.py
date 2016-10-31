import matplotlib.pyplot as plt
import numpy as np
import seaborn
import matplotlib.gridspec as gridspec

seaborn.set_style("white")

def predict_plot_size_viz_data(data, row_names, col_names):
    if isinstance(data, list): data = np.array(data)
    row_width = max(map(len, map(str, row_names)))
    col_height = max(map(len, map(str, col_names)))
    height = data.shape[0]
    width = data.shape[1]
    return height, width, row_width, col_height * np.cos(np.pi/4)

def viz_data_raw(data, ax=None, row_names=None,
                 col_names=None, cmap=None, title=None,
                 labelsize=None):
    if isinstance(data, list): data = np.array(data)
    if ax is None: ax = plt.gca()
    if cmap is None: cmap = "YlGn"
    if row_names is None: row_names = range(data.shape[0])
    if col_names is None: col_names = range(data.shape[1])
    if labelsize is None: labelsize = 12
    height, width, _, _ = predict_plot_size_viz_data(data, row_names, col_names)
        
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
        ax.set_yticklabels(row_names, ha='right', size=labelsize)
#         ax.set_yticklabels(row_names, ha='right')
        
    xticklabelsize = size - width/3.
    if col_names is not None:
        ax.set_xticks(range(data_normed.shape[1]))
        ax.set_xticklabels(col_names, rotation=45, rotation_mode='anchor',
                            ha='left', size=labelsize)
#         ax.set_xticklabels(col_names, rotation=45, rotation_mode='anchor',
#                             ha='left')
    # Hack to set grids off-center
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in ax.get_yticks()][1:], minor='true')
    ax.grid(True, which='minor')
    
    return ax

def viz_data(data, ax=None, row_names=None, col_names=None, 
             cmap=None, title=None, savefile=None, labelsize=None):
    """
    Vizualize heterogeneous data.
    Standardize data across columns.
    Ignore nan values (plotted white)
    """
    if savefile is None: savefile = "viz_data_foo.png"
    if row_names is None: row_names = range(data.shape[0])
    if col_names is None: col_names = range(data.shape[1])

    ax = viz_data_raw(data, ax, row_names, col_names, cmap, title, labelsize)
    height, width, _, _ = predict_plot_size_viz_data(data, row_names, col_names)
    
    fig = ax.get_figure()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    fig.set_tight_layout(True)
    if savefile:
        fig.savefig(savefile)
    return ax

def viz_view_raw(view, ax=None, row_names=None, col_names=None, labelsize=None):
    """ 
    Order rows according to clusters and draw line between clusters.
    Visualize this using imshow with two colors only.
    """
    if ax is None: ax = plt.gca()
    if isinstance(row_names, list): row_names = np.array(row_names)
    if isinstance(col_names, list): col_names = np.array(col_names)
    
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
        row_names = row_names[row_indexes]

    output_cols = data_dict.keys()

    if col_names is None:
        col_names = output_cols
    elif len(col_names) == len(view.X.values()):
        col_names = col_names[output_cols]

    # Plot clustered data 
    ax = viz_data_raw(clustered_data, ax, row_names, col_names, labelsize=labelsize)
    
    # Plot lines between clusters
    for bd in cluster_boundaries:
        ax.plot(
            [-0.5, clustered_data.shape[1]-0.5], [bd+0.5, bd+0.5],
            color='magenta', linewidth=3)

    return ax

def viz_view(view, ax=None, row_names=None, col_names=None, 
             savefile=None, labelsize=None):
    """ 
    Order rows according to clusters and draw line between clusters.
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

    ax = viz_view_raw(view, ax, row_names, col_names, labelsize=labelsize)
    
    height, width, _, _ = predict_plot_size_viz_data(data_arr, row_names, col_names)
    
    fig = ax.get_figure()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    fig.set_tight_layout(True)
    if savefile:
        fig.savefig(savefile)
    return ax

def viz_state(state, row_names=None, col_names=None, savefile=None, labelsize=None):
    """ 
    For each each view call viz_view as defined above. 
    Plot each view next to each other.
    Do not use gridspec
    """
    data_arr = np.array(state.X.values()).T

    if row_names is None: row_names = range(data_arr.shape[0])
    if col_names is None: col_names = range(data_arr.shape[1])
    if savefile is None: savefile = "state_foo.png"

    D = data_arr.shape[1]
    views = state.views.keys()
    
    # Get the subplot widths
    view_widths = []
    for view in views:
        data_view = np.array(get_view_data(state.views[view]).values()).T
        height, width, row_width, col_height = predict_plot_size_viz_data(
            data_view, row_names, col_names)
        view_widths.append(width)

    wspace = row_width/1.3
    all_wspace = wspace * len(views)
    right_margin = col_height / 2.
    fig_width = sum(view_widths + [all_wspace] + [right_margin]) * 1.
    fig_height = height
    Z = 5.

    fig = plt.figure(figsize=(fig_width/Z, fig_height/Z))
    # Create grid for subplots 

    def get_ax_param_list(view_widths, wspace, fig_width, right_margin):
        assert np.allclose(fig_width, sum(view_widths + [wspace] * len(
            view_widths) + [right_margin]))
        ymin = 0.05
        dy = 0.65
        dx_list = [w / fig_width for w in view_widths]
        norm_wspace = wspace / fig_width
        xmin_list = [norm_wspace]
        for i, _ in enumerate(view_widths[:-1]):
            xmin_list.append(xmin_list[i] + dx_list[i] + norm_wspace)
        
        return [(xmin, ymin, dx, dy) for (xmin, dx) in zip(xmin_list, dx_list)]
        

    # Plot data for each viewx
    ax_param_list = get_ax_param_list(view_widths, wspace,
                                      fig_width, right_margin)
    ax_list = []
    for (i, view) in enumerate(views):
        ax_params = ax_param_list[i]
        ax_list.append(fig.add_axes(ax_params))
        ax_list[-1] = viz_view_raw(
            state.views[view], ax_list[-1], row_names, col_names, labelsize=labelsize)
        
    # gs.tight_layout(fig)
    fig.subplots_adjust(right=0.95)
    if savefile:
        fig.savefig(savefile)
    return ax_list
 

def viz_state_old(state, row_names=None, col_names=None, savefile=None):
    """ 
    For each each view call viz_view as defined above. 
    Plot each view next to each other.
    """
    data_arr = np.array(state.X.values()).T

    if row_names is None: row_names = range(data_arr.shape[0])
    if col_names is None: col_names = range(data_arr.shape[1])
    if savefile is None: savefile = "state_foo.png"

    D = data_arr.shape[1]
    views = state.views.keys()
    view_assigns = [state.Zv()[i] for i in range(D)]    
    
    # Get the subplot widths
    view_widths = []
    view_heights = []
    for view in views:
        data_view = np.array(get_view_data(state.views[view]).values()).T
        height, width, row_width, col_height = predict_plot_size_viz_data(data_view, row_names, col_names)
        view_widths.append(width)
        view_heights.append(height)

    # compute xlabelsize adaptively

    # wspace = row_width/2. # compute wspace as a function of the number of max(map(len, col_names))
    # norm_factor = 4.
    wspace = row_width
    all_wspace = wspace * (len(views)-1)
    fig_width = sum(view_widths + [all_wspace]) / 4
    fig_height = view_heights[0] / 4

    fig = plt.figure(figsize=(fig_width, fig_height))
    # Create grid for subplots 
    gs = gridspec.GridSpec(
        1, len(views), width_ratios=view_widths)
    
    # Figsize in pixels
    # [ ] Force width space in pixels
    # [ ] Force column width in pixels

    # Plot data for each view
    ax_list = []
    for (i, view) in enumerate(views):
        ax_list.append(fig.add_subplot(gs[i]))
    
        ax_list[-1] = viz_view_raw(state.views[view], ax_list[-1],
                                   row_names, col_names)
        # ax_list[-1].set_xlabel("View %d" % (view), size='x-large')
        
    # gs.tight_layout(fig)
    plt.subplots_adjust(top=0.7, bottom=0.05, right=0.95, left=0.05)
    if savefile:
        fig.savefig(savefile)
    return ax_list
 

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
