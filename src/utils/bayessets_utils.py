import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sb

from cgpm.mixtures.view import View
from cgpm.crosscat.state import State
from cgpm.bayessets import bayes_sets as bs
from cgpm.utils import general as gu
from timeit import default_timer as timer

OUTDIR = 'test/resources/out/'
ANIMALSPATH = 'tests/resources/animals.csv'
RNG = gu.gen_rng(0)

def init_array(size):
    return np.empty(size) * np.nan

def order_dataset_by_score(query, score_function, csv_path):
    """
    Loads dataset from csv_path and orders it according to the
    score as outputted from score_function.
 
    Parameters
    ----------
    query          - array-like of size (n x D)
    score_function - function(target, query) -> float
    csv_path       - string with path to dataset .csv file
    
    Returns
    -------
    ordered_indices - pd.DataFrame(cols=['name', 'score']}
           DataFrame with ordered indices from dataset and respective scores
    """
    df = pd.read_csv(csv_path, index_col='name')
    # Drop Nan
    null_ix = df[df.isnull().any(axis=1)].index
    df.drop(null_ix, axis=0, inplace=True)
    # Evaluate score
    data = df.values
    query_array = df.ix[query].values
    score_array, _, _ = score_dataset(data, query_array, score_function)
    ordered_indices = pd.DataFrame({'name': df.index,
                                    'score': score_array.flatten()})
    ordered_indices.sort_values('score', ascending=False, inplace=True)
    ordered_indices.index = range(1, len(ordered_indices) + 1)
    return ordered_indices

def init_view_state(iters):
    data = pd.read_csv(ANIMALSPATH, index_col="name").values
    D = len(data[0])
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(
        X,
        cctypes=['bernoulli']*D,
        outputs=[1000] + outputs,
        rng=RNG)
    state = State(
        data[:, 0:D],
        outputs=outputs,
        cctypes=['bernoulli']*D,
        rng=RNG)
    if iters > 0:
        view.transition(N=iters)
        state.transition(N=iters)
    return view, state


def animal_experiment(query, iters):
    """
    Scores animal dataset according to query and 
    returns 
    """
    # Load data 
    t_start = timer()
    view, state = init_view_state(iters)
    comparison_df = comparison_experiment(
        query, ANIMALSPATH, view, state)
    t_end = timer()
    comp_time = t_end - t_start

    return comparison_df, comp_time
   
def comparison_experiment(query, csv_path, view, state):
    """
    Orders dataset from path_csv for three different scoring
    functions: 
    1. parametric score, 
    2. DP Mixture score,
    3. Crosscat score,
    with respect to query.

    Parameters
    ----------
    query     - array-like of size (n x D)
    csv_path  - string with path to dataset .csv file    
    view      - CGPM object for the DP Mixture
    state     - CGPM object for Crosscat

    Returns
    -------
    Dataframe containing both the ordered indices from the three functions 
    and their respective scores.
    """
    print "\n\nComputing Parametric Logscore"
    indices_parametric = order_dataset_by_score(
        query, bs.binary_logscore, csv_path)
    indices_parametric.columns = ['name_parametric', 'score_parametric']
    
    print "\n\nComputing DP Mixture Logscore"
    indices_view = order_dataset_by_score(
        query, view.generative_logscore, csv_path)
    indices_view.columns = ['name_dpmbb', 'score_dpmbb']
    
    print "\n\nComputing Crosscat Logscore"
    indices_crosscat = order_dataset_by_score(
        query, state.generative_logscore, csv_path)
    indices_crosscat.columns = ['name_crosscat', 'score_crosscat']
    
    return pd.concat([indices_parametric, indices_view, indices_crosscat], axis=1)

def score_histograms(df, query):
    fig, ax = plt.subplots(1,3, sharey=True, sharex=True)
    sb.distplot(df['score_parametric'], rug=True, bins=15,
                ax=ax[0], vertical=True, norm_hist=False)
    ax[0].set_xlabel('Parametric Bayes Sets (frequency)', fontsize=13)
    ax[0].set_ylabel('score', fontsize=15)
    sb.distplot(df['score_dpmbb'], rug=True, bins=15, ax=ax[1], vertical=True)
    ax[1].set_xlabel('DPMixture Bayes Sets (frequency)', fontsize=13)
    ax[1].set_ylabel('')
    sb.distplot(df['score_crosscat'], rug=True, bins=15, ax=ax[2], vertical=True)
    ax[2].set_xlabel('Crosscat Bayes Sets (frequency)', fontsize=13)
    ax[2].set_ylabel('')
    fig.suptitle('Query: %s' %(", ".join(query),), fontsize=15)
    return fig, ax

def sample_beta_sparse(D=20, rng=RNG):
    beta_1 = .1
    beta_0 = .1
    return rng.beta(beta_1, beta_0, size=D)

def binary_linear_plot(data):
    fig, ax = plt.subplots(1, 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_ylabel('Entities, N = %s' % (data.shape[0],))
    ax.set_xlabel('Features, D = %s' % (data.shape[1],))
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.imshow(data, cmap='Greys', interpolation='nearest',
              vmin=0, vmax=1)
    return fig, ax

def binary_square_plot(data):
    n_squared = data.shape[0]
    n = np.sqrt(n_squared)
    assert n.is_integer()
    sqr_data = np.reshape(data, (n, n)) 
    fig, ax = plt.subplots(1, 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.imshow(sqr_data, cmap='Greys', interpolation='nearest',
              vmin=0, vmax=1)
    return fig, ax

def plot_dataset(dataset, save, name):
    binary_linear_plot(dataset)
    if save:
        plt.savefig(OUTDIR + name + '.png')

def score_dataset(dataset, query, score_function):
    """
    Computes score_function(x, query) for each datapoint
    x in dataset.

    Returns:
    --------
    score_array     - array-like (<float)
                array with the scores of each point in the dataset
    sorted_indices  - array-like (<int>)
                indices of the dataset after sorting according to score_array 
    sorted_dataset  - array-like
                dataset after sorting according to score_array 
    """
    
    num_datapoints = dataset.shape[0]    
    score_array = init_array(size=(num_datapoints, 1))
    for i in range(num_datapoints):
        score_array[i] = score_function(dataset[i, :], query)
    
    sorted_indices = np.argsort(score_array, axis=0).T[0][::-1]
    sorted_dataset = dataset[sorted_indices, :]

    return score_array, sorted_indices, sorted_dataset

def experimental_setup(dataset, query, save, n, name, score_function=None):
    """
    Parameters:
    -----------
    dataset        - array-like
    query          - array-like
    save           - Boolean
    n              - int,
                     Number of query points
    name           - str,
                     Name to save the figure
    score_function - 'binary_logscore', 'binary_score', 'dpmbb_logscore'
    """
    if score_function is None: score_function = bs.binary_logscore
    num_datapoints = dataset.shape[0]    

    score_array, sorted_indices, sorted_dataset = score_dataset(
        dataset, query, score_function)

    fig, ax = binary_linear_plot(sorted_dataset)
    ax.set_yticks(range(num_datapoints))
    dec_labels = ["%.2f" % (f,) for f in score_array[sorted_indices]]
    ax.set_yticklabels(dec_labels, size=8)
    ax.set_ylabel('log-score, |Query| = %d' % (n, ))

    if save:
        plt.savefig(OUTDIR + name + '_scored_q%d.png' % (n,))
    plt.close("all")
