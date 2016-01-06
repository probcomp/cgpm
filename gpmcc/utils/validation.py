# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import gpmcc.utils.config as cu

def validate_metadata(metadata):
    # FIXME: fill in
    pass

def validate_cctypes(cctypes):
    for cctype in cctypes:
        if not cu.valid_distgpm(cctype):
            raise ValueError("Invalid cctype %s. Valid values: %s" % \
                (str(cctype), str(cu.all_distgpms())))

def validate_data(X):
    if not (isinstance(X, list) or isinstance(X, np.ndarray)):
        raise TypeError("Data should be a list or numpy array.")
    if not isinstance(X[0], np.ndarray):
        raise TypeError("All entries in data should by numpy arrays.")
    else:
        num_rows = X[0].shape[0]
    for x in X:
        if not isinstance(x, np.ndarray):
            raise TypeError("All entries in data should by numpy arrays.")
        if x.shape[0] != num_rows:
            raise ValueError("All columns should have the same number of rows.")
