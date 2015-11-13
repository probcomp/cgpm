# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy

cctypes_list = ['normal', 'normal_uc','binomial','multinomial','lognormal',
    'poisson','vonmises','vonmises_uc']

def validate_metadata(metadata):
    # FIXME: fill in
    pass

def validate_cctypes(cctypes):
    for cctype in cctypes:
        if cctype not in cctypes_list:
            raise ValueError("Invalid cctype %s. Valid values: %s" % \
                (str(cctype), str(cctypes_list)))

def validate_data(X):
    if not isinstance(X, list):
        raise TypeError("Data should be a list.")
    if not isinstance(X[0], numpy.ndarray):
        raise TypeError("All entries in data should by numpy arrays.")
    else:
        num_rows = X[0].shape[0]
    for x in X:
        if not isinstance(x, numpy.ndarray):
            raise TypeError("All entries in data should by numpy arrays.")
        if x.shape[0] != num_rows:
            raise ValueError("All columns should have the same number of rows.")
