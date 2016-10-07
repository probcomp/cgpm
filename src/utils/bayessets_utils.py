import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sb

from cgpm.bayessets.mvbernoulli import IndependentBernoulli
from collections import namedtuple
from cgpm.bayessets import bayes_sets as bs

TwiceTwentyCoins = namedtuple('TwiceTwentyCoins', [
    'name', 'twenty_coins_first', 'twenty_coins_second', 'data_first',
    'data_second', 'all_data', 'shuffled_data', 'shuffled_indices'])

Coins = namedtuple('Coins', [
    'name', 'coin_sets', 'data', 'all_data',
    'shuffled_data', 'shuffled_indices'])
OUTDIR = '../out/'
RNG = np.random.RandomState(0)
DATA = '../data/'

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

def animal_experiment(query, score_function):
    """
    Scores animal dataset according to query and 
    returns 
    """
    # Load data 
    csv_path = DATA + 'animaldata.csv'
    return order_dataset_by_score(query, score_function, csv_path)
   
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

def generate_twice_twenty_coins(name, p1, p2):
    """
    Generate two 20-dimensional Bernoulli and returns the 
    named tuple corresponding to them.
    """
    twenty_coins_first = IndependentBernoulli(params={
        'p': p1}, rng=RNG)

    twenty_coins_second = IndependentBernoulli(params={
        'p': p2}, rng=RNG)        

    # Generate toy data
    data_first = twenty_coins_first.simulate_array(N=25)
    data_second = twenty_coins_second.simulate_array(N=25)
    all_data = np.vstack((data_first, data_second))
    
    # Shuffle data
    shuffled_indices = np.random.permutation(range(all_data.shape[0]))
    shuffled_data = all_data[shuffled_indices, :]

    return TwiceTwentyCoins(
        name, twenty_coins_first, twenty_coins_second, data_first,
        data_second, all_data, shuffled_data, shuffled_indices)

def generate_ttc_gradthresh():
    return generate_twice_twenty_coins(
        name='ttc_gradthresh', p1=np.linspace(0, 1, 20), p2=[.8]*8 + [.2]*12)

def generate_ttc_concentrated():
    return generate_twice_twenty_coins(
        name='ttc_concentrated', p1=[.9]*5 + [.1]*15, p2=[.1]*15 + [.9]*5)

def sample_beta_sparse(D=20, rng=RNG):
    beta_1 = .1
    beta_0 = .1
    return rng.beta(beta_1, beta_0, size=D)

def generate_three_sparse_coin_sets(name='default_three_sparse'):
    coin_sets = [IndependentBernoulli(params={
        'p': sample_beta_sparse(20)}, rng=RNG) for i in range(3)]
    
    data = [coin_sets[i].simulate_array(N=20) for i in range(3)]
    all_data = np.vstack((data[i] for i in range(3)))

    shuffled_indices = np.random.permutation(range(all_data.shape[0]))
    shuffled_data = all_data[shuffled_indices, :]

    return Coins(
        name, coin_sets, data, all_data, shuffled_data, shuffled_indices)

def generate_three_crosscat_coin_sets(name='default_three_crosscat'):
    three_coin_sets_left = [IndependentBernoulli(params={
        'p': sample_beta_sparse(10)}, rng=RNG) for i in range(3)]
    two_coins_sets_right = [IndependentBernoulli(params={
        'p': sample_beta_sparse(10)}, rng=RNG) for i in range(2)]
    coin_sets = [three_coin_sets_left, two_coins_sets_right]

    data_left = [coin_sets[0][i].simulate_array(N=20) for i in range(3)]
    all_data_left = np.vstack((data_left[i] for i in range(3)))
    data_right = [coin_sets[1][i].simulate_array(N=30) for i in range(2)]
    all_data_right = np.vstack((data_right[i] for i in range(2)))
    data = [data_left, data_right]
    all_data = np.hstack((all_data_left, all_data_right))

    shuffled_indices = np.random.permutation(range(all_data.shape[0]))
    shuffled_data = all_data[shuffled_indices, :]

    return Coins(
        name, coin_sets, data, all_data, shuffled_data, shuffled_indices)

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

def plot_ttc(ttc=None, save=None):
    if ttc is None: ttc = generate_ttc_gradthresh()
    if save is None: save = False
    # Visualize toy data separately
    ttc.twenty_coins_first.plot_samples(N=25)
    if save:
        plt.savefig(OUTDIR + ttc.name + '_first_samples.png')
    # plt.show()

    ttc.twenty_coins_second.plot_samples(N=25)
    if save:
        plt.savefig(OUTDIR + ttc.name + '_second_samples.png')
    # plt.show()

    # Visualize mixed toy_data
    binary_linear_plot(ttc.all_data)
    if save:
        plt.savefig(OUTDIR + ttc.name + '_all_samples.png')
    # plt.show()

    binary_linear_plot(ttc.shuffled_data)
    if save:
        plt.savefig(OUTDIR + ttc.name + '_shuffled_samples.png')
    # plt.show()

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

def experiment_ttc(n=None, ttc=None, save=None, score_function=None):
    if ttc is None: ttc = generate_ttc_gradthresh()
    if save is None: save = False
    if n is None: n = 4
    if score_function is None: score_function = bs.binary_logscore
    plot_ttc(ttc)

    name = ttc.name 
    query = ttc.data_first[:n, :]
    dataset = ttc.shuffled_data

    plot_dataset(dataset, save, name + "_dataset")
    plot_dataset(query, save, name + "_query")

    experimental_setup(dataset, query, save, n, name, score_function)

def experiment_three_sparse_coin_sets(n=None, tcs=None, save=None, score_function=None):
    if tcs is None: tcs = generate_three_sparse_coin_sets()
    if save is None: save = False
    if n is None: n = 4
    name = tcs.name + "_" + score_function.__name__
    query = np.vstack((tcs.data[i][:n/2, :] for i in range(2)))
    dataset = tcs.all_data

    plot_dataset(dataset, save, name + "_dataset")
    plot_dataset(query, save, name + "_query")

    experimental_setup(dataset, query, save, n, name, score_function)
    
def experiment_three_crosscat_coin_sets(n=None, txs=None, save=None, score_function=None):
    if txs is None: txs = generate_three_crosscat_coin_sets()
    if save is None: save = False

    n = 4
    name = txs.name
    dataset = txs.all_data
    query = dataset[[0, 15, 29, 45], :]

    plot_dataset(dataset, save, name)
    plot_dataset(query, save, name + "_query")

    experimental_setup(dataset, query, save, n, name, score_function)
