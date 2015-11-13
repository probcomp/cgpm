from baxcat import cc_state
from baxcat.utils import cc_test_utils as tu
from baxcat.utils import cc_sample_utils as su
from baxcat.utils import cc_general_utils as utils

import sys
import random
import numpy
import pylab

_all_kernels = ['column_hypers','view_alphas','row_z','state_alpha','column_z']

def forward_sample(X, n_iters, Zv=None, Zrcv=None, n_grid=30, n_chains=1, ct_kernel=0):
    total_iters = n_chains*n_iters
    n_cols = len(X)
    cctypes = ['normal']*n_cols
    distargs = [None]*n_cols
    forward_samples = dict()
    stats = []
    i = 0
    for chain in range(n_chains):
        forward_samples[chain] = []
        for itr in range(n_iters):
            i += 1
            state = cc_state.cc_state(X, cctypes, distargs, Zv=Zv, Zrcv=Zrcv, n_grid=n_grid, ct_kernel=ct_kernel)
            Y = su.resample_data(state)
            forward_samples[chain].append(Y)
            stats.append(get_data_stats(Y, state))
            string = "\r%1.2f  " % (i*100.0/float(total_iters))
            sys.stdout.write(string)
            sys.stdout.flush()

    return stats, forward_samples

def posterior_sample(X, n_iters, kernels=_all_kernels, Zv=None, Zrcv=None, n_grid=30, n_chains=1, ct_kernel=0):
    n_cols = len(X)
    cctypes = ['normal']*n_cols
    distargs = [None]*n_cols
    stats = []
    posterior_samples = dict()
    i = 0.0;
    total_iters = n_chains*n_iters
    for chain in range(n_chains):
        state = cc_state.cc_state(X, cctypes, distargs, Zv=Zv, Zrcv=Zrcv, n_grid=n_grid, ct_kernel=ct_kernel)
        Y = su.resample_data(state)
        posterior_samples[chain] = Y
        for _ in range(n_iters):
            state.transition(kernel_list=kernels)
            Y = su.resample_data(state)
            stats.append(get_data_stats(Y, state))
            posterior_samples[chain].append(Y)
            i += 1.0
            string = "\r%1.2f  " % (i*100.0/float(total_iters))
            sys.stdout.write(string)
            sys.stdout.flush()
            
    return stats, posterior_samples

def get_data_stats(X, state):
    # Only get's stats for first column
    stats = dict()
    stats['std'] = numpy.std(X[0])
    stats['mean'] = numpy.mean(X[0])
    stats['col_alpha'] = state.alpha
    stats['row_alpha'] = state.views[0].alpha
    stats['hyper_s'] = state.dims[0].hypers['s']
    stats['hyper_m'] = state.dims[0].hypers['m']
    stats['hyper_r'] = state.dims[0].hypers['r']
    stats['hyper_nu'] = state.dims[0].hypers['nu']

    return stats

def prep_stats_for_plotting(stats):
    pstats = dict(
        std=[],
        mean=[],
        hyper_m=[],
        hyper_s=[],
        hyper_r=[],
        hyper_nu=[],
        col_alpha=[],
        row_alpha=[],
        )
    for stat in stats:
        pstats['mean'].append(stat['mean'])
        pstats['std'].append(stat['std'])
        pstats['hyper_m'].append(stat['hyper_m'])
        pstats['hyper_s'].append(stat['hyper_s'])
        pstats['hyper_r'].append(stat['hyper_r'])
        pstats['hyper_nu'].append(stat['hyper_nu'])
        pstats['row_alpha'].append(stat['row_alpha'])
        pstats['col_alpha'].append(stat['col_alpha'])

    return pstats

def pp_plot(f, p, nbins):
    uniqe_vals_f = list(set(f))
    uniqe_vals_p = list(set(p))

    combine = uniqe_vals_f
    combine.extend(uniqe_vals_p)
    combine = list(set(combine))

    if len(uniqe_vals_f) > nbins:
        bins = nbins
    else:
        bins = sorted(combine)
        bins.append( bins[-1]+bins[-1]-bins[-2] )

    ff, edges = numpy.histogram(f, bins=bins, density=True)
    fp, _ = numpy.histogram(p, bins=edges, density=True)

    Ff = numpy.cumsum(ff*(edges[1:]-edges[:-1]))
    Fp = numpy.cumsum(fp*(edges[1:]-edges[:-1]))

    pylab.plot([0,1],[0,1],c='red', lw=2)
    pylab.plot(Ff,Fp, c='black', lw=1)
    pylab.xlim([0,1])
    pylab.ylim([0,1])

def binned_hist_pair(x_plots, stat_f, stat_p, index, nbins, stat_name, log_scale_x=False, log_scale_y=False):

    uniqe_vals_f = list(set(stat_f))
    uniqe_vals_p = list(set(stat_p))
    combine = uniqe_vals_f
    combine.extend(uniqe_vals_p)
    combine = list(set(combine))
    if len(combine) > nbins:
        bins = nbins
    else:
        bins = sorted(combine)
        bins.append( bins[-1]+bins[-1]-bins[-2] )

    ax1 = pylab.subplot(2,x_plots,index)
    _, bins_f, _ = pylab.hist(stat_f, bins=bins, normed=True, histtype='stepfilled', alpha=.7,color='blue')
    pylab.title('%s (forward)' % stat_name)
    pylab.xlim([min(bins_f), max(bins_f)])
    y1 = ax1.get_ylim()

    if log_scale_x:
        ax1.set_xscale('log')
    if log_scale_y:
        ax1.set_yscale('log')

    ax2 = pylab.subplot(2,x_plots,x_plots+index)
    pylab.hist(stat_p, bins=bins_f, normed=True, histtype='stepfilled',alpha=.7,color='red')
    pylab.title('%s (posterior)'% stat_name)
    pylab.xlim([min(bins_f), max(bins_f)])
    y2 = ax1.get_ylim()

    if log_scale_x:
        ax2.set_xscale('log')
    if log_scale_y:
        ax2.set_yscale('log')

    if y2[1] > y1[1]:
        ax1.set_ylim(y2)
    else:
        ax2.set_ylim(y1)

def plot_stats(stats_f, stats_p, hbins=30):

    stats_f = prep_stats_for_plotting(stats_f)
    stats_p = prep_stats_for_plotting(stats_p)

    # data
    print("plotting data stats")
    fig = pylab.figure(num=None, facecolor='w', edgecolor='k',frameon=False, tight_layout=True)

    ax = pylab.subplot(2,2,1)
    _, bins_f, _ = pylab.hist(stats_f['mean'], bins=numpy.linspace(-5,5), normed=True, histtype='stepfilled')
    pylab.title('col 0 data mean (forward)')
    ax = pylab.subplot(2,2,3)
    pylab.hist(stats_p['mean'], bins=bins_f, normed=True, histtype='stepfilled')
    pylab.title('col 0 data mean (posterior)')

    
    ax = pylab.subplot(2,2,2)
    _, bins_f, _ = pylab.hist(stats_f['std'], bins=numpy.linspace(0,5), normed=True, histtype='stepfilled')
    pylab.title('col 0 data std (forward)')
    ax = pylab.subplot(2,2,4)
    pylab.hist(stats_p['std'], bins=bins_f, normed=True,histtype='stepfilled')
    pylab.title('col 0 data std (posterior)')


    # hypers
    pylab.show()
    print("plotting prior stats")

    fig = pylab.figure(num=None, facecolor='w', edgecolor='k',frameon=False, tight_layout=True)

    pylab.clf()
    ax = pylab.subplot(2,3,1)
    pp_plot(stats_f['hyper_m'], stats_p['hyper_m'], hbins)
    pylab.xlabel('forward')
    pylab.ylabel('posterior')
    pylab.title('hyper m')

    ax = pylab.subplot(2,3,2)
    pp_plot(stats_f['hyper_s'], stats_p['hyper_s'], hbins)
    pylab.xlabel('forward')
    pylab.ylabel('posterior')
    pylab.title('hyper s')

    ax = pylab.subplot(2,3,3)
    pp_plot(stats_f['hyper_r'], stats_p['hyper_r'], hbins)
    pylab.xlabel('forward')
    pylab.ylabel('posterior')
    pylab.title('hyper r')

    ax = pylab.subplot(2,3,4)
    pp_plot(stats_f['hyper_nu'], stats_p['hyper_nu'], hbins)
    pylab.xlabel('forward')
    pylab.ylabel('posterior')
    pylab.title('hyper nu')

    ax = pylab.subplot(2,3,5)
    pp_plot(stats_f['col_alpha'], stats_p['col_alpha'], hbins)
    pylab.xlabel('forward')
    pylab.ylabel('posterior')
    pylab.title('hyper col alpha')

    ax = pylab.subplot(2,3,6)
    pp_plot(stats_f['row_alpha'], stats_p['row_alpha'], hbins)
    pylab.xlabel('forward')
    pylab.ylabel('posterior')
    pylab.title('hyper row alpha')

    pylab.show()

    fig = pylab.figure(num=None, facecolor='w', edgecolor='k',frameon=False, tight_layout=True)

 
    binned_hist_pair(6, stats_f['hyper_m'], stats_p['hyper_m'], 1, hbins, 'col 0 hyper mu')
    binned_hist_pair(6, stats_f['hyper_s'], stats_p['hyper_s'], 2, hbins, 'col 0 hyper s', log_scale_x=True)
    binned_hist_pair(6, stats_f['hyper_r'], stats_p['hyper_r'], 3, hbins, 'col 0 hyper r', log_scale_x=True)
    binned_hist_pair(6, stats_f['hyper_nu'], stats_p['hyper_nu'], 4, hbins, 'col 0 hyper nu', log_scale_x=True)
    binned_hist_pair(6, stats_f['col_alpha'], stats_p['col_alpha'], 5, hbins, 'colum crp alpha', log_scale_x=True)
    binned_hist_pair(6, stats_f['row_alpha'], stats_p['row_alpha'], 6, hbins, 'view_0 row crp alpha', log_scale_x=True)


    pylab.show()
def _contstruct_transition_list(t_col_z, t_row_z, t_state_alpha, t_view_alpha, t_col_hyper):
    kernels = [];
    if t_col_hyper:
        kernels.append('column_hypers')
    if t_state_alpha:
        kernels.append('state_alpha')
    if t_view_alpha:
        kernels.append('view_alphas')
    if t_col_z:
        kernels.append('column_z')
    if t_row_z:
        kernels.append('row_z')
    
    

    return kernels

def _construct_state_args(t_col_z, t_row_z, n_rows, n_cols):
    kernels = [];

    if not t_row_z:
        t_col_z = False
        Zrcv = [[0]*n_rows]
    else:
        Zrcv = None

    if not t_col_z:
        Zv = numpy.zeros(n_cols, dtype=int)
    else:
        Zv = None

    return Zv, Zrcv


if __name__ == '__main__':
    random.seed(0)
    numpy.random.seed(0)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rows', default=10, type=int)
    parser.add_argument('--num_cols', default=10, type=int)
    parser.add_argument('--num_chains', default=1, type=int)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--no_col_z', default=True, action="store_false")
    parser.add_argument('--no_row_z', default=True, action="store_false")
    parser.add_argument('--no_state_alpha', default=True, action="store_false")
    parser.add_argument('--no_col_hyper', default=True, action="store_false")
    parser.add_argument('--no_view_alpha', default=True, action="store_false")
    parser.add_argument('--n_grid', default=50, type=int)
    parser.add_argument('--ct_kernel', default=0, type=int)

    args = parser.parse_args()
    #
    n_rows = args.num_rows
    n_cols = args.num_cols
    n_chains = args.num_chains
    n_iters = args.num_iters
    t_col_z = args.no_col_z;
    t_row_z = args.no_row_z;
    t_state_alpha = args.no_state_alpha;
    t_col_hyper = args.no_col_hyper;
    t_view_alpha = args.no_view_alpha;
    n_grid = args.n_grid
    ct_kernel = args.ct_kernel

    # python geweke_test.py --num_rows 40 --num_cols 1 --num_iters 1000 --num_chains 1 --no_col_z --no_row_z

    kernels = _contstruct_transition_list(t_col_z, t_row_z, t_state_alpha, t_view_alpha, t_col_hyper);
    Zv, Zrcv = _construct_state_args(t_col_z, t_row_z, n_rows, n_cols)

    # X = [numpy.random.normal(size=(n_rows)) for _ in range(n_cols)]
    X = [numpy.ones(n_rows) * float('nan') for _ in range(n_cols)]

    print("Generating forward samples")
    forward_sample_stats, fs = forward_sample(X, n_iters, Zv, Zrcv, n_grid, n_chains, ct_kernel)
    print(" ")
    print("Generating posterior samples")
    postertior_sample_stats, ps = posterior_sample(X, n_iters, kernels, Zv, Zrcv, n_grid, n_chains, ct_kernel)
    print(" ")
    print("Plotting samples")

    plot_stats(forward_sample_stats , postertior_sample_stats, hbins=n_grid)



    