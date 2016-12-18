import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgpm.mixtures.view import View

def load_csv():
    df = pd.read_csv('tests/resources/iris_pc_noname.csv')
    return df

def load_view():
    df = load_csv()
    X = {c: df[['flower_pc1', 'flower_pc2']].values[:, c] for c in range(2)}
    Zr = df['class']
    outputs = [1000, 0, 1]
    view = View(X, outputs, cctypes=['normal', 'normal'], Zr=Zr)
    
    return view

def test_search_row_146():
    view = load_view()

    query = {146: {}}
    # import pudb; pudb.set_trace()
    view.relevance_search(query)
    pass

def test_make_rowid_contiguous():
    view = load_view()
    
    rowid_lst = [0, 146]
    query = {row: {} for row in rowid_lst}

    for row in rowid_lst:
        view.unincorporate(row)

    test_out = view._make_rowid_contiguous(query)
    exp_out = query

    assert exp_out == test_out

def test_incorporate_unincorporate_last_row():
    view = load_view()

    rowid = 150
    row = {0: 1.21904446391, 1: -0.23199466788400003, 1000: 0}

    view.incorporate(rowid, row)
    assert len(view.X[0]) == 151

    view.unincorporate(rowid)
