import bayes_sets as bs
import numpy as np
import experiment as exp
import matplotlib
matplotlib.use("Agg")
    
def test_binary_score():
    bs.binary_score(target=[0, 1, 1, 0], query=np.eye(4))
    bs.binary_score(target=[0, 1, 1, 0], query=np.eye(4),
                    hypers={'alpha': [.5]*4, 'beta': [.5]*4})

def test_binary_logscore():
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

def test_logscore_synthetic():
    ld = exp.generate_ttc_gradthresh()
    query = ld.data_first
    dataset = ld.shuffled_data

    num_datapoints = ld.shuffled_data.shape[0]    
    for i in range(num_datapoints):
        assert not np.isnan(bs.binary_logscore(dataset[i, :], query))

def test_crash_dpmbb_logscore()

def test_plot_ttc():
    exp.plot_ttc(exp.generate_ttc_gradthresh())
    # exp.plot_ttc(exp.generate_ttc_left_right())

def test_experiment_ttc():
    exp.experiment_ttc(n=10, ttc=exp.generate_ttc_gradthresh())
    exp.experiment_ttc(n=10, ttc=exp.generate_ttc_concentrated())
