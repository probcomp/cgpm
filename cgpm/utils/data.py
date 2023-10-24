# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import numpy as np
import pandas as pd

from cgpm.utils import config as cu


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
            'university': {
                'mit': 0,
                'harvard': 1
                },
            'country': {
                'usa': 0,
                'nepal': 1,
                'lebanon': 2
                }
            }
        where 'k' for 'country' has been extracted from the dataset.

    >>> S = cgpm.crosscat.state.State(D, cctypes=cctypes, distargs=distargs)
    """
    dataframe.replace('NA', np.nan, inplace=True)
    D = []
    cctypes, distargs = [], []
    valmap = dict()
    columns = []
    outputs = []
    for column, stattype, index in schema:
        if stattype == 'ignore':
            continue
        X = dataframe[column]
        columns.append(column)
        cctypes.append(stattype)
        outputs.append(index)
        distargs.append(None)
        # XXX Should check for is_numeric!
        if stattype in ['bernoulli', 'categorical']:
            mapping = build_valmap(X)
            X = X.replace(mapping)
            valmap[column] = mapping
            if stattype == 'bernoulli':
                assert len(mapping) == 2
            else:
                # Did user specify categorical mapping?
                dist, k_user = cu.parse_distargs([column])
                if k_user == [None]:
                    distargs[-1] = {'k': len(mapping)}
                else:
                    assert len(mapping) <= k_user
        D.append(X)
    T = np.asarray(D).T
    assert len(cctypes) == len(distargs) == len(columns)
    assert len(columns)  == T.shape[1]
    return T, outputs, cctypes, distargs, valmap, columns


def build_valmap(column):
    uniques = [u for u in sorted(column.unique()) if not pd.isnull(u)]
    return {u:k for k,u in enumerate(uniques)}


def dummy_code(x, discretes):
    """Dummy code a vector of covariates x for ie regression.

    Parameters
    ----------
    x : list
        List of data. Categorical values must be integer starting at 0.
    discretes : dict{int:int}
        discretes[i] is the number of discrete categories in x[i].

    Returns
    -------
    xd : list
        Dummy coded version of x as list.

    Example
    -------
    >>> dummy_code([12.1, 3], {1:5})
    [12.1, 0, 0, 0, 1]
    # Note only 4 dummy codes since all 0s indicates cateogry 4.
    """
    if len(discretes) == 0:
        return list(x)
    def as_code(i, val):
        if i not in discretes:
            return [val]
        if float(val) != int(val):
            raise TypeError('Discrete value must be integer: {},{}'.format(x,i))
        k = discretes[i]
        if not 0 <= val < k:
            raise ValueError('Discrete value not in {0..%s}: %d.'% (k-1, val))
        r = [0]*(k-1)
        if val < k-1:
            r[int(val)] = 1
        return r
    xp = [as_code(i, val) for i, val in enumerate(x)]
    return list(itertools.chain.from_iterable(xp))
