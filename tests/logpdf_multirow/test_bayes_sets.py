from cgpm.bayessets import bayes_sets as bs
import numpy as np
from cgpm.utils import bayessets_utils as bu
import matplotlib
matplotlib.use("Agg")
    
def test_crash_binary_score():
    bs.binary_score(target=[0, 1, 1, 0], query=np.eye(4))
    bs.binary_score(target=[0, 1, 1, 0], query=np.eye(4),
                    hypers={'alpha': [.5]*4, 'beta': [.5]*4})

def test_crash_binary_logscore():
    bs.binary_logscore(target=[0, 1, 1, 0], query=np.eye(4))
    bs.binary_logscore(target=[0, 1, 1, 0], query=np.eye(4),
                       hypers={'alpha': [.5]*4, 'beta': [.5]*4})

def test_score_coherence():
    score_1 = bs.binary_score(target=[0, 1, 1, 0], query=np.eye(4))
    logscore_1 = bs.binary_logscore(target=[0, 1, 1, 0], query=np.eye(4))
    assert np.allclose(np.exp(logscore_1), score_1)

    score_2 = bs.binary_score(target=[0, 1, 1, 0], query=np.eye(4),
                              hypers={'alpha': [.5]*4, 'beta': [.5]*4})
    logscore_2 = bs.binary_logscore(target=[0, 1, 1, 0], query=np.eye(4),
                                    hypers={'alpha': [.5]*4, 'beta': [.5]*4})
    assert np.allclose(np.exp(logscore_2), score_2)

def test_two_column_binary_score():
    # P(x | alpha = beta = 1.) for x in [[0,0], [0,1], [1,0], [1,1]]
    p_target = np.array([.25]*4)
    
    # P(x | y = [1,1], alpha = beta = 1)
    p_target_posterior = np.array([1./9, 2./9, 2./9, 4./9])

    math_score = p_target_posterior / p_target
    assert np.allclose(
        bs.binary_score(target=[0,0], query=[[1, 1]]), math_score[0])
    assert np.allclose(
        bs.binary_score(target=[0,1], query=[[1, 1]]), math_score[1])
    assert np.allclose(
        bs.binary_score(target=[1,0], query=[[1, 1]]), math_score[2])
    assert np.allclose(
        bs.binary_score(target=[1,1], query=[[1, 1]]), math_score[3])

def test_two_column_binary_logscore():
    # P(x | alpha = beta = 1.) for x in [[0,0], [0,1], [1,0], [1,1]]
    p_target = np.array([.25]*4)
    
    # P(x | y = [1,1], alpha = beta = 1)
    p_target_posterior = np.array([1./9, 2./9, 2./9, 4./9])

    math_score = p_target_posterior / p_target
    assert np.allclose(
        bs.binary_logscore(target=[0,0], query=[[1, 1]]), np.log(math_score[0]))
    assert np.allclose(
        bs.binary_logscore(target=[0,1], query=[[1, 1]]), np.log(math_score[1]))
    assert np.allclose(
        bs.binary_logscore(target=[1,0], query=[[1, 1]]), np.log(math_score[2]))
    assert np.allclose(
        bs.binary_logscore(target=[1,1], query=[[1, 1]]), np.log(math_score[3]))

def test_eval_betabernoulli_prior():
    # P(x | alpha = beta = 1.) for x in [[0,0], [0,1], [1,0], [1,1]]
    p_target = np.array([.25]*4)
    assert np.allclose(bs.eval_betabernoulli([0, 0]), p_target[0])
    assert np.allclose(bs.eval_betabernoulli([0, 1]), p_target[1])
    assert np.allclose(bs.eval_betabernoulli([1, 0]), p_target[2])
    assert np.allclose(bs.eval_betabernoulli([1, 1]), p_target[3])

def test_eval_betabernoulli_posterior():
    # P(x | y = [1,1], alpha = beta = 1)
    p_target_posterior = np.array([1./9, 2./9, 2./9, 4./9])
    assert np.allclose(
        bs.eval_betabernoulli([0, 0], [[1, 1]]), p_target_posterior[0])
    assert np.allclose(
        bs.eval_betabernoulli([1, 0], [[1, 1]]), p_target_posterior[1])
    assert np.allclose(
        bs.eval_betabernoulli([0, 1], [[1, 1]]), p_target_posterior[2])
    assert np.allclose(
        bs.eval_betabernoulli([1, 1], [[1, 1]]), p_target_posterior[3])

def test_two_column_binary_new_score():
    # P(x | alpha = beta = 1.) for x in [[0,0], [0,1], [1,0], [1,1]]
    p_target = np.array([.25]*4)

    # P(x | y = [1,1], alpha = beta = 1)
    p_target_posterior = np.array([1./9, 2./9, 2./9, 4./9])

    math_score = p_target_posterior / p_target
    assert np.allclose(
        bs.binary_score_alternative(
        target=[0, 0], query=[[1, 1]]), math_score[0])
    assert np.allclose(
        bs.binary_score_alternative(
        target=[0, 1], query=[[1, 1]]), math_score[1])
    assert np.allclose(
        bs.binary_score_alternative(
        target=[1, 0], query=[[1, 1]]), math_score[2])
    assert np.allclose(
        bs.binary_score_alternative(
        target=[1, 1], query=[[1, 1]]), math_score[3])
