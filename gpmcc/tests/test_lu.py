import baxcat.utils.cc_test_utils as tu
import baxcat.cc_state
import numpy

from baxcat.utils import cc_legacy_utils as lu

n_rows = 100
view_weights = numpy.ones(2)/2
cluster_weights = [numpy.ones(3)/3.0, numpy.ones(2)/2.0]
cctypes = ['normal']*5
distargs = [None]*5
separation = [.7, .9]

T, Zv, Zc, dims = tu.gen_data_table(
                    n_rows, 
                    view_weights, 
                    cluster_weights, 
                    cctypes, 
                    distargs, 
                    separation, 
                    return_dims=True)


state = cc_state.cc_state(T, cctypes, distargs)
state.transition(N=10)

M_c, X_L, X_D = lu.get_legacy_metadata(state)

Tcc = T[0]
for i in range(1,len(T)):
    Tcc = numpy.vstack( (Tcc, T[i]) )
Tcc = numpy.transpose(Tcc) 

# make sure the data came out right
for i in range(len(T)):
    assert numpy.all(T[i] == Tcc[:,i])

state_b = lu.construct_state_from_legacy_metadata(Tcc, M_c, X_L, X_D)

engine = lu.BaxCatEngine()

# initialize with one state
X_L, X_D = engine.initialize(M_c, [], Tcc)
assert isinstance(X_L, dict)

# initialize with multiple states
X_L_list, X_D_list = engine.initialize(M_c, [], Tcc, n_chains=5)
assert len(X_L_list) == 5

# analyze on single state
X_L, X_D = engine.analyze(M_c, Tcc, X_L, X_D)
assert isinstance(X_L, dict)

# analyze on multiple states
X_L_list, X_D_list = engine.analyze(M_c, Tcc, X_L_list, X_D_list)
assert len(X_L_list) == 5

Q = [(0,0),(n_rows-1,0),(0,2)]

# simple predictive samples on one state
samples_single = engine.simple_predictive_sample(M_c, X_L, X_D, None, Q, n=1)

# simple predictive samples on one state
samples_multi = engine.simple_predictive_sample(M_c, X_L_list, X_D_list, None, Q, n=1)

# An acutual T, M_c, X_L, and X_D from CrossCat
T = [
[0.5377, 0.6715, -0.1022, -1.0891],
[1.8339, -1.2075, -0.2414, 0.0326],
[-2.2588, 0.7172, 0.3192, 0.5525],
[0.8622, 1.6302, 0.3129, 1.1006],
[0.3188, 0.4889, -0.8649, 1.5442],
[-1.3077, 1.0347, -0.0301, 0.0859],
[-0.4336, 0.7269, -0.1649, -1.4916],
[0.3426, -0.3034, 0.6277, -0.7423],
[3.5784, 0.2939, 1.0933, -1.0616],
[2.7694, -0.7873, 1.1093, 2.3505],
[-1.3499, 0.8884, -0.8637, -0.6156],
[3.0349, -1.1471, 0.0774, 0.7481],
[0.7254, -1.0689, -1.2141, -0.1924],
[-0.0631, -0.8095, -1.1135, 0.8886],
[0.7147, -2.9443, -0.0068, -0.7648],
[-0.2050, 1.4384, 1.5326, -1.4023],
[-0.1241, 0.3252, -0.7697, -1.4224],
[1.4897, -0.7549, 0.3714, 0.4882],
[1.4090, 1.3703, -0.2256, -0.1774],
[1.4172, -1.7115, 1.1174, -0.1961]]
M_c = {'idx_to_name': {'1': 1, '0': 0, '3': 3, '2': 2}, 'column_metadata': [{'code_to_value': {}, 'value_to_code': {}, 'modeltype': 'normal_inverse_gamma'}, {'code_to_value': {}, 'value_to_code': {}, 'modeltype': 'normal_inverse_gamma'}, {'code_to_value': {}, 'value_to_code': {}, 'modeltype': 'normal_inverse_gamma'}, {'code_to_value': {}, 'value_to_code': {}, 'modeltype': 'normal_inverse_gamma'}], 'name_to_idx': {0: 0, 1: 1, 2: 2, 3: 3}}
X_L = {"column_partition": {"assignments": [0, 0, 1, 0], "counts": [3, 1], "hypers": {"alpha": 1.6624757922855753}}, "column_hypers": [{"mu": 1.2435199999999995, "s": 16.561350387365785, "r": 10.985605433061178, "fixed": 0.0, "nu": 10.985605433061178}, {"mu": 0.5628166666666676, "s": 4.401977628278786, "r": 6.034176336545162, "fixed": 0.0, "nu": 3.662517131494462}, {"mu": 0.2508066666666666, "s": 2.1550883657338895, "r": 4.941771544836058, "fixed": 0.0, "nu": 4.472135954999579}, {"mu": 0.17330999999999974, "s": 0.6173129362279323, "r": 4.472135954999579, "fixed": 0.0, "nu": 2.7144176165949063}], "view_state": [{"column_component_suffstats": [[{"sum_x": 3.4840999999999998, "sum_x_squared": 8.180372450000002, "N": 2.0}, {"sum_x": 7.791700000000002, "sum_x_squared": 34.96084873, "N": 15.0}, {"sum_x": 2.0159000000000002, "sum_x_squared": 7.292474590000001, "N": 3.0}], [{"sum_x": -3.7316, "sum_x_squared": 9.28874378, "N": 2.0}, {"sum_x": 3.5104999999999995, "sum_x_squared": 15.453310089999999, "N": 15.0}, {"sum_x": -0.9277000000000001, "sum_x_squared": 3.0985343500000004, "N": 3.0}], [{"sum_x": 1.5856999999999997, "sum_x_squared": 6.109769289999999, "N": 2.0}, {"sum_x": -3.5568, "sum_x_squared": 14.814981179999997, "N": 15.0}, {"sum_x": 0.6067, "sum_x_squared": 0.24678081000000004, "N": 3.0}]], "row_partition_model": {"counts": [2, 15, 3], "hypers": {"alpha": 1.490976045823099}}, "column_names": [0, 1, 3]}, {"column_component_suffstats": [[{"sum_x": -2.7469, "sum_x_squared": 2.57829803, "N": 3.0}, {"sum_x": 1.1094999999999997, "sum_x_squared": 1.5667028699999999, "N": 6.0}, {"sum_x": 0.8969000000000003, "sum_x_squared": 0.5423900900000002, "N": 3.0}, {"sum_x": 1.0805000000000002, "sum_x_squared": 1.24953501, "N": 3.0}, {"sum_x": -1.1366999999999998, "sum_x_squared": 1.4800295699999997, "N": 2.0}, {"sum_x": 1.5326, "sum_x_squared": 2.34886276, "N": 1.0}, {"sum_x": 0.22839999999999994, "sum_x_squared": 1.9433569, "N": 2.0}]], "row_partition_model": {"counts": [3, 6, 3, 3, 2, 1, 2], "hypers": {"alpha": 2.999468681960427}}, "column_names": [2]}]}
X_D = [[1, 2, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1], [2, 1, 1, 1, 6, 3, 1, 2, 6, 1, 0, 4, 4, 0, 3, 5, 0, 2, 1, 3]]

state_c = lu.construct_state_from_legacy_metadata(T, M_c, X_L, X_D)
