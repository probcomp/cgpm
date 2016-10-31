import cgpm.utils.render_utils as ru
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from random import choice
from string import ascii_uppercase

from cgpm.crosscat.state import State
from cgpm.crosscat.engine import Engine
from cgpm.mixtures.view import View


RESOURCES = 'tests/resources/'
OUT = 'tests/resources/out/'
RNG = np.random.RandomState(7)

# Define Data
animals = pd.read_csv(RESOURCES + "animals_cc.csv", index_col=0)
animal_values = animals.values
animal_names = animals.index.values
animal_features = animals.columns.values

# Load engines
with open(
        RESOURCES + 'animals_cc_60models_100iters_seed7.pkl', 'rb') as fileptr:
    engine_cc = Engine.from_pickle(fileptr)

with open(
        RESOURCES + 'animals_lovecat_64models_1000iters_seed7.pkl', 'rb') as fileptr:
    engine_lovecat = Engine.from_pickle(fileptr)


# # HELPERS # #
def engine_filename(engine, name, iters=None):
    
    if iters is None:
        if engine.get_state(0).iterations:
            iters = engine.get_state(0).iterations['rows']
        else:
            iters = 0
    
    models = engine.num_states()
    filename = '%s_%dmodels_%diters_seed7' % (name, models, iters)
    return filename
    
def render_states_to_disk(engine, name, iters=None):
    filename = engine_filename(engine, name, iters)
    
    filepath = 'out/' + filename
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    num_states = engine.num_states()
    for i in range(num_states):
        savefile = filepath + "/state_%d" %(i,)
        state = engine.get_state(i)
        ru.viz_state(state, row_names=animal_names, col_names=animal_features,savefile=savefile);
    plt.close()

# # TESTS # #

def test_render_many_views():
    savefile = OUT + "test_render_many_views.png"
    state = engine_cc.get_state(12)
    ru.viz_state(state, row_names=animal_names,
                 col_names=animal_features, savefile=savefile)

def test_render_couple_views():
    savefile = OUT + "test_render_couple_views.png"
    state = engine_lovecat.get_state(32)
    ru.viz_state(state, row_names=animal_names,
                 col_names=animal_features, savefile=savefile)

def test_render_one_view():
    savefile = OUT + "test_render_one_view.png"
    state = engine_cc.get_state(0)
    ru.viz_state(state, row_names=animal_names,
                 col_names=animal_features, savefile=savefile)

def test_render_many_views_nogs():
    savefile = OUT + "test_render_many_views_nogs.png"
    state = engine_cc.get_state(12)
    ru.viz_state_nogs(state, row_names=animal_names,
                      col_names=animal_features, savefile=savefile)

def test_render_couple_views_nogs():
    savefile = OUT + "test_render_couple_views_nogs.png"
    state = engine_lovecat.get_state(32)
    ru.viz_state_nogs(state, row_names=animal_names,
                      col_names=animal_features, savefile=savefile)

def test_render_one_view_nogs():
    savefile = OUT + "test_render_one_view_nogs.png"
    state = engine_cc.get_state(0)
    ru.viz_state_nogs(state, row_names=animal_names,
                      col_names=animal_features, savefile=savefile)
