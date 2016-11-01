import numpy as np
import pandas as pd
import seaborn

from cgpm.crosscat.state import State
from cgpm.crosscat.engine import Engine
from cgpm.utils import bayessets_utils as bu
from cgpm.bayessets import bayes_sets as bs
from cgpm.utils import datasearch_utils as du
from cgpm.utils import render_utils as ru

PKLDIR = 'tests/resources/pkl/'
DATADIR = 'tests/resources/data/'
ANIMALSPATH = DATADIR + 'animals_cc.csv'
OUT = 'tests/resources/out/'

# Data Variables
animals = pd.read_csv(ANIMALSPATH, index_col='name')
animal_values = animals.values
animal_names = animals.index.values
animal_features = animals.columns.values

# View State Variables
seed = 7
RNG = np.random.RandomState(seed)
with open(PKLDIR + 'animals_view_lovecat_64models_1000iters_seed7.pkl', 'rb') as fileptr:
    engine = Engine.from_pickle(fileptr, rng=RNG)
    
view_state =  engine.get_state(12)

eagle_row = animals.ix['Eagle'].values
wolf_row = animals.ix['Wolf'].values
finch_row = animals.ix['Finch'].values

def test_bug1():
    assert (view_state.generative_logscore(wolf_row, [eagle_row]) !=
            view_state.generative_logscore(finch_row, [eagle_row]))

def test_bug2():
    # Compute the logpdf of wolf and finch
    logpdf_wolf = view_state.logpdf(-1, du.list_to_dict(wolf_row))
    logpdf_finch = view_state.logpdf(-1, du.list_to_dict(finch_row))

    # Compute the logpdf multirow of wolf and finch
    logpdfm_wolf_eagle = view_state.logpdf_multirow(-1, 
                                             du.list_to_dict(wolf_row),
                                             du.list_to_dict([eagle_row]))

    logpdfm_wolf_finch = view_state.logpdf_multirow(-1, 
                                             du.list_to_dict(wolf_row),
                                             du.list_to_dict([finch_row]))

    logpdfm_finch_eagle = view_state.logpdf_multirow(-1, 
                                             du.list_to_dict(finch_row),
                                             du.list_to_dict([eagle_row]))

    assert (logpdfm_wolf_eagle - logpdf_wolf !=
            logpdfm_finch_eagle - logpdf_finch)
    assert logpdfm_wolf_finch != logpdfm_wolf_eagle
