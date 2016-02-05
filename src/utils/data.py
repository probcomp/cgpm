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
import pandas as pd

from gpmcc.utils import config as cu

def parse_schema(schema, dataframe):
    """Apply a schema to a dataframe, and return variables to construct State.

    Parameters
    ----------
    schema : list(tuple)
        A list of tuples, where each tuple is ('column', 'stattype'). The values
        of 'stattype' are either DistirbutionGpms, or 'ignore'. For categorical
        datatypes, it is permitted to specify the number of components distarg
        by 'categorical(k=7)' although make sure the number of components is
        correct; if unspecified, the number of components will be estimated from
        the dataset.
    dataframe : pd.DataFrame
        Dataframe containing the dataset to parse according to the schema. All
        missing values must be 'NA' or np.nan -- otherwise very bad things will
        happen.

    Returns
    -------
    D : np.array
        Data matrix that gpmcc can ingest.
    cctypes : list<str>
        List of cctype strings that gpmcc can ingest.
    distargs : list<dict>
        Distargs for the cctypes, according to the schema.
    valmap : dict<str->dict>
        For Bernoulli or categorical columns, strings are converted to integer
        values in [0..k]. valmap['column'] gives the mapping from strings to
        integers for such columns. Needed for reference only, not for gpmcc.
    columns : list<str>
        List of column names, where columns[i] is the ith column of D. Needed
        for reference only, not for gpmcc.

    Example
    -------
    >>> dataframe = pd.read_csv('dataset.csv')
    >>> schema = [('id','ignore'), ('age','normal'), ('gender','bernoulli'),
    ...     ('university','categorical(k=2)'), ('country','categorical')]
    >>> D, cctypes, distargs, valmap, columns = parse_schema(dataframe, schema)

    -   D will be the dataset as np array.
    -   cctypes = ['normal', 'bernoulli', 'categorical', 'categorical']
    -   distargs = [None, None, {'k':2}, {'k':3}]
    -   valmap = {
            'university': {'mit':0, 'harvard':1},
            'country': {'usa':0, 'nepal':1, 'lebanon':2}
            }
        where 'k' for 'country' has been extracted from the dataset.

    >>> S = gpmcc.state.State(D, cctypes, distargs=distargs)
    """
    dataframe.replace('NA', np.nan, inplace=True)
    D = []
    cctypes, distargs = [], []
    valmap = dict()
    columns = []
    for column, stattype in schema:
        if stattype == 'ignore':
            continue
        X = dataframe[column]
        columns.append(column)
        cctypes.append(stattype)
        distargs.append(None)
        if stattype in ['bernoulli', 'categorical']:
            mapping = dict()
            k = 0
            for val in X.unique():
                if not pd.isnull(val):
                    mapping[val] = k
                    k += 1
            X = X.replace(mapping)
            valmap[column] = mapping
            if stattype == 'bernoulli':
                assert len(mapping) == 2
            else:
                # Did user specify categorical mapping?
                _, k = cu.parse_distargs([column])
                if k == [None]:
                    distargs[-1] = {'k':len(mapping)}
                else:
                    assert k >= len(mapping)
        D.append(X)
    T = np.asarray(D).T
    assert len(cctypes) == len(distargs) == len(columns)
    assert len(columns)  == T.shape[1]
    return T, cctypes, distargs, valmap, columns
