import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import pandas as pd

from cgpm.mixtures.view import View
from cgpm.crosscat.state import State
from cgpm.utils import general as gu
from cgpm.utils import bayessets_utils as bu

from timeit import default_timer as timer

"""
This tests should be run from the main folder cgpm/
"""

OUT = 'tests/resources/out/'
ANIMALSPATH = 'tests/resources/animals.csv'

def load_animals():
    return pd.read_csv(ANIMALSPATH, index_col="name")

@pytest.fixture(params=[(iters, dim) 
                        for iters in [0, 50]
                        for dim in [1, 85]])
def cgpms(request):
    N = request.param[0]
    D = request.param[1]
    rng = gu.gen_rng(0)
    data = load_animals().values
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(
        X,
        cctypes=['bernoulli']*D,
        outputs=[1000] + outputs,
        rng=rng)
    state = State(
        data[:, 0:D],
        outputs=outputs,
        cctypes=['bernoulli']*D,
        rng=rng)
    if N > 0:
        view.transition(N=N)
        state.transition(N=N)
    return view, state

def test_comparison_experiment(cgpms):
    view, state = cgpms
    evidence = ['dalmatian']
    t_start = timer()
    comparison_df = bu.comparison_experiment(
        evidence, ANIMALSPATH, view, state)
    t_end = timer()
    comp_time = t_end - t_start

    D = len(state.dims())
    iterations = state.iterations
    if not iterations:
        N = 0
    else:
        N = max(iterations.values())
    comparison_df.to_csv(OUT + "bs_animal_comparison_%ddim_%diterations.csv"
                         % (D, N))

    fig, ax = bu.score_histograms(comparison_df, evidence)
    fig.suptitle(''' Query: %s.\n
                     Computation time: %.2f ''' % (evidence, comp_time))
    fig.savefig(OUT + "scored_histograms_%ddims_%diterations" % (D, N),
                dpi=900)


