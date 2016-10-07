from math import gamma
from collections import namedtuple
import numpy as np
from numpy import log
from dpmbb import DPMBetaBernoulli

# def _assert_type_score(x_dot, big_j, big_dc):

VARIABLE_NAMES = [
    'x_dot', 'big_j', 'big_dc', 'big_n', 'alpha', 'beta',
    'alpha_tilde', 'beta_tilde', 'small_c', 'small_q']
PaperVariables = namedtuple('PaperVariables', VARIABLE_NAMES)

def assert_no_nan(array):
    assert np.all([not np.isnan(x_i) for x_i in array]) 

def get_paper_vars(target, query, hypers):
    """
    Get variables and hyperparameters that 
    are necessary for the binary score from
    the Bayesian Sets paper

    Parameters
    ----------
    target - array-like of shape (1,D)
    query  - array-like of shape (n,D)
    hypers - dict{alpha: positive array-like of shape (1,D),
                  beta: positive array-like of shape (1,D)}, optional
    Returns:
    --------
    PaperVariables - namedtuple with variable names
    """

    x_dot = np.array(target)
    big_j = x_dot.shape[0]
    big_dc = np.array(query)
    big_n = big_dc.shape[0]

    if not hypers: hypers = {}
    alpha = np.array(hypers.get('alpha', [1.] * big_j))
    beta = np.array(hypers.get('beta', [1.] * big_j))
    alpha_tilde = alpha + big_dc.sum(axis=0)
    beta_tilde = beta + big_n - big_dc.sum(axis=0)

    small_c = sum([
        log(alpha[j] + beta[j]) - log(alpha[j] + beta[j] + big_n) +
        log(beta_tilde[j]) - log(beta[j])
        for j in range(big_j)])
    small_q = np.array([
        log(alpha_tilde[j]) - log(alpha[j]) -
        log(beta_tilde[j]) + log(beta[j])
        for j in range(big_j)])
    
    assert not np.isnan(small_c)
    assert_no_nan(small_q)
    assert big_dc.shape[1] == big_j
    assert alpha.shape[0] == beta.shape[0] == big_j

    return PaperVariables(
        x_dot, big_j, big_dc, big_n, alpha, beta,
        alpha_tilde, beta_tilde, small_c, small_q)

def binary_score(target, query, hypers=None):
    """ Calculate Bayesian Sets score for binary data
    of target data point wrt to query, given beta hyperparameters.

    Bayesian Sets - Gharamani & Heller, 2006.
    'Our algorithm uses a modelbased concept of a cluster and ranks
    items using a score which evaluates the marginal probability that
    each item belongs to a cluster containing the query items.'

    Parameters
    ----------
    target - array-like of shape (1,D)
    query  - array-like of shape (n,D)
    hypers - dict{alpha: positive array-like of shape (1,D),
                  beta: positive array-like of shape (1,D)}, optional
    """
    v = get_paper_vars(target, query, hypers)

    score = 1
    for j in range(v.big_j):
        factor_left = (v.alpha[j] + v.beta[j]) / (
            v.alpha[j] + v.beta[j] + v.big_n)
        factor_center = (v.alpha_tilde[j] / v.alpha[j]) ** v.x_dot[j]
        factor_right = (v.beta_tilde[j] / v.beta[j]) ** (1 - v.x_dot[j])
        score *= factor_left * factor_center * factor_right

    return score

def binary_logscore(target, query, hypers=None):
    """
    Log of binary score (see binary_score)
    """
    v = get_paper_vars(target, query, hypers)

    logscore = v.small_c + sum([
        v.small_q[j] * v.x_dot[j] for j in range(v.big_j)])

    assert not np.isnan(logscore)
    return logscore

def dpmbb_logscore(target, query, rng=None):
    """
    Uses the DP Mixture of Beta Bernoullis CGPM to compute:
    
    logscore = log(target|query) - log(target)
    
    Parameters:
    -----------
    target - array-like of shape (1,D)
    query  - array-like of shape (n,D)
    """
    target_dict = list_to_dict(target)
    query_dict = list_to_dict(query)

    crp_alpha = 1
    dim = len(target)
    bb_hypers = [.1, .1]
    if rng is None: rng = np.random.RandomState(0)
    dpmbb = DPMBetaBernoulli(crp_alpha, dim, bb_hypers, rng)

    logp_target = dpmbb.logpdf(target_dict)
    logp_conditional = dpmbb.logpdf_multirow(target_dict, query_dict)
    
    logscore = logp_conditional - logp_target
    return logscore

def cgpm_logscore(target, query, cgpm, rng=None):
    """
    Uses the CGPM interface to compute:
    
    logscore = log(target|query) - log(target)
    
    Parameters:
    -----------
    target - array-like of shape (1,D)
    query  - array-like of shape (n,D)
    cgpm   - CGPM object (equipped with logpdf_multirow method)
    """
    if not isinstance(target, dict):
        target_dict = list_to_dict(target)
    if not isinstance(query, dict):
        query_dict = list_to_dict(query)

    logp_target = cgpm.logpdf(target_dict)
    logp_conditional = cgpm.logpdf_multirow(target_dict, query_dict)
    
    logscore = logp_conditional - logp_target
    return logscore
