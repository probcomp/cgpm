import numpy as np
import pytest

from cgpm.crosscat.state import State

def gen_data():
    rng = np.random.RandomState(0)
    return rng.rand(3, 2)

def gen_cctypes():
    return ['normal'] * 2

def test_cctypes_is_not_optional():
    with pytest.raises(ValueError):
        state = State(X=gen_data())
        # ValueError: Specify a cctype!

def test_only_X_cctypes_necessary():
    state = State(
        X=gen_data(),
        cctypes=gen_cctypes())
    # Passes

def test_Zv_is_not_a_list():
    with pytest.raises(AttributeError):
        state = State(
            X=gen_data(),
            cctypes=gen_cctypes(),
            Zv=[0, 0])
        # AttributeError: 'list' object has no attribute 'iteritems'

def test_Zv_is_a_dict():
    state = State(
        X=gen_data(),
        cctypes=gen_cctypes(),
        Zv={0: 0, 1: 0})

def test_Zrv_is_not_a_list():
    with pytest.raises(AttributeError):
        state = State(
            X=gen_data(),
            cctypes=gen_cctypes(),
            Zv={0: 0, 1: 0},
            Zrv=[[0,0]])
        # AttributeError: 'list' object has no attribute 'keys'

def test_Zrv_is_a_dict():
    state = State(
        X=gen_data(),
        cctypes=gen_cctypes(),
        Zv={0: 0, 1: 0},
        Zrv={0: [0, 0, 0]})
