import cgpm.utils.render_utils as ru
import matplotlib.pyplot as plt 
import numpy as np
import os 

from random import choice
from string import ascii_uppercase

from cgpm.crosscat.state import State
from cgpm.mixtures.view import View

PKLDIR = 'tests/resources/pkl/'
OUT = 'tests/resources/out/'
RNG = np.random.RandomState(7)
# Define Data

test_dataset_dpmm = [  # data3
    [True,  False, False, False, False, True],
    [True,  False, False, False, False, True],
    [True,  False, False, False, False, True],
    [True,  False, False, False, False, True],
    [True,  False, False, False, False, True],
    [True,  False, False, False, False, True],
    [True,  False, False, False, False, True],
    [False, False, True,  True,  True,  False],
    [False, False, True,  True,  True,  False],
    [False, False, True,  True,  True,  False],
    [False, False, True,  True,  True,  False],
    [False, False, True,  True,  True,  False]
];

test_dataset_with_distractors = [ # data2 
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True,  False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, True,  True,  True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, True,  True,  True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, True,  True,  True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, True,  True,  True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, True,  True,  True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
];

X1 = np.array(test_dataset_dpmm)
X2 = RNG.normal(10, 5, size=[12, 6])
test_dataset_mixed = np.hstack((X1, X2))
test_dataset_mixed_nan = np.vstack((test_dataset_mixed, [np.nan]*12))
test_dataset_wide = np.hstack((test_dataset_mixed,
                               test_dataset_mixed,
                               test_dataset_mixed))
test_dataset_tall = np.vstack((test_dataset_mixed,
                               test_dataset_mixed,
                               test_dataset_mixed))

# Initialize DPMM and CrossCat models for the data above

def init_view_state(data, iters, cctypes):
    if isinstance(data, list): data = np.array(data)
    D = len(data[0])
    outputs = range(D)
    X = {c: data[:, i].tolist() for i, c in enumerate(outputs)}
    view = View(
        X,
        cctypes=cctypes,
        outputs=[1000] + outputs,
        rng=RNG)
    state = State(
        data[:, 0:D],
        outputs=outputs,
        cctypes=cctypes,
        rng=RNG)    
    if iters > 0:
        view.transition(iters)
        state.transition(iters)
    return view, state

# # Helpers # #
def string_generator(N=1, length=10):
    return [(''.join(choice(ascii_uppercase)
                     for i in range(length))) for i in range(N)]

# view1, state1 = init_binary_view_state(test_dataset_dpmm, 50)
# view2, state2 = init_binary_view_state(test_dataset_with_distractors, 50)
view3, state3 = init_view_state(
    test_dataset_mixed_nan, 25, ['bernoulli']*6 + ['normal']*6)
row_names_test = string_generator(12, 10)
col_names_test = string_generator(6, 7)
row_names3 = string_generator(13, 10)
col_names3 = string_generator(12, 7)

def test_viz_data():
    savefile = OUT + "test_viz_data.png"
    ru.viz_data(test_dataset_mixed_nan, savefile=savefile)

def test_viz_data_with_names():
    savefile = OUT + "test_viz_data_with_names.png"
    ru.viz_data(test_dataset_dpmm, row_names=row_names_test,
                col_names=col_names_test, savefile=savefile)

def test_viz_wide_data():
    savefile = OUT + "test_viz_wide_data.png"
    ru.viz_data(test_dataset_wide, savefile=savefile)

def test_viz_tall_data():
    savefile = OUT + "test_viz_tall_data.png"
    ru.viz_data(test_dataset_tall, savefile=savefile)

def test_viz_view():
    savefile = OUT + "test_viz_view.png"
    ru.viz_view(view3, savefile=savefile)

def test_viz_view_with_names():
    savefile = OUT + "test_viz_view_with_names.png"
    ru.viz_view(view3, row_names=row_names3,
                col_names=col_names3, savefile=savefile)
    
def test_viz_state():
    savefile = OUT + "test_viz_state.png"
    ru.viz_state(state3, savefile=savefile)
    
def test_viz_state_with_names():
    savefile = OUT + "test_viz_state_with_names.png"
    ru.viz_state(state3, row_names=row_names3,
                 col_names=col_names3, savefile=savefile)
    
def test_viz_state_curlette_no_colnames():
    filepath = PKLDIR + 'state_curlette.pkl'
    if not os.path.isfile(filepath):
        raise Warning("%s file does not exist. Aborting", (filepath,))
        pass 
    else:
        with open(filepath, 'rb') as fileptr:
            state = State.from_pickle(fileptr)
        savefile = OUT + "test_viz_state_curlette_no_colnames.png"
        ru.viz_state(state, savefile=savefile)

def test_viz_state_curlette_with_colnames():
    filepath = PKLDIR + 'state_curlette.pkl'
    if not os.path.isfile(filepath):
        raise Warning("%s file does not exist. Aborting", (filepath,))
        pass 
    else:
        with open(PKLDIR + 'state_curlette.pkl', 'rb') as fileptr:
            state = State.from_pickle(fileptr)
        savefile = OUT + "test_viz_state_curlette_with_colnames.png"
        ru.viz_state(state, savefile=savefile, col_names=[str(i) for i in range(35)])
