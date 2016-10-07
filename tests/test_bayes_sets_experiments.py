import pytest
import numpy as np
import pandas as pd

from cgpm.mixtures.view import View
from cgpm.crosscat.state import State
from cgpm.utils import general as gu
from cgpm.utils import bayessets_utils as bu

"""
This tests should be run from the main folder cgpm/
"""

OUT = 'tests/resources/out/'
ANIMALSPATH = 'tests/resources/animals.csv'

def load_animals():
    return pd.read_csv(ANIMALSPATH, index_col="name")

@pytest.fixture(params=[0, 10])
def cgpms(request):
    rng = gu.gen_rng(0)
    data = load_animals().values
    D = len(data[0])
    outputs = range(D)
    N = request.param
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(
        X,
        cctypes=['bernoulli']*D,
        outputs=[1000] + outputs,
        rng=rng)
    state = State(
        data,
        outputs=outputs,
        cctypes=['bernoulli']*D,
        rng=rng)
    if N > 0:
        view.transition(N=N)
        state.transition(N=N)
    return view, state

def test_comparison_experiment(cgpms):
    view, state = cgpms
    evidence = ['grizzly bear', 'killer whale', 'lion']
    comparison_df = bu.comparison_experiment(
        evidence, ANIMALSPATH, view, state)

    comparison_df.to_csv(OUT + "bs_animal_comparison.csv")

    fig, ax = bu.score_histograms(comparison_df, evidence)
    fig.savefig(OUT + "scored_histograms")
