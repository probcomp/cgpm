# -*- coding: utf-8 -*-

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

import numpy as np

def valid_types():
    """Returns a list of colors for plotting."""
    return ['beta_uc', 'binomial', 'lognormal', 'multinomial', 'normal',
        'normal_uc', 'poisson','vonmises', 'vonmises_uc']

def validate_metadata(metadata):
    # FIXME: fill in
    pass

def validate_cctypes(cctypes):
    for cctype in cctypes:
        if cctype not in valid_types():
            raise ValueError("Invalid cctype %s. Valid values: %s" % \
                (str(cctype), str(valid_types())))

def validate_data(X):
    if not (isinstance(X, list) or isinstance(X, np.ndarray)):
        raise TypeError("Data should be a list.")
    if not isinstance(X[0], np.ndarray):
        raise TypeError("All entries in data should by numpy arrays.")
    else:
        num_rows = X[0].shape[0]
    for x in X:
        if not isinstance(x, np.ndarray):
            raise TypeError("All entries in data should by numpy arrays.")
        if x.shape[0] != num_rows:
            raise ValueError("All columns should have the same number of rows.")
