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

import re

from datetime import datetime

import importlib


cctype_class_lookup = {
    'bernoulli'         : ('cgpm.exponentials.bernoulli', 'Bernoulli'),
    'beta_uc'           : ('cgpm.exponentials.beta_uc', 'BetaUC'),
    'categorical'       : ('cgpm.exponentials.categorical', 'Categorical'),
    'crp'               : ('cgpm.exponentials.crp', 'Crp'),
    'exponential'       : ('cgpm.exponentials.exponential', 'Exponential'),
    'geometric'         : ('cgpm.exponentials.geometric', 'Geometric'),
    'linear_regression' : ('cgpm.regressions.linreg', 'LinearRegression'),
    'lognormal'         : ('cgpm.exponentials.lognormal', 'Lognormal'),
    'normal'            : ('cgpm.exponentials.normal', 'Normal'),
    'normal_trunc'      : ('cgpm.exponentials.normal_trunc', 'NormalTrunc'),
    'poisson'           : ('cgpm.exponentials.poisson', 'Poisson'),
    'random_forest'     : ('cgpm.regressions.forest', 'RandomForest'),
    'vonmises'          : ('cgpm.exponentials.vonmises', 'Vonmises'),
}

def timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def colors():
    """Returns a list of colors."""
    return ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown',
        'black', 'pink']

def cctype_class(cctype):
    """Return class object for initializing a named GPM (default normal)."""
    if not cctype: raise ValueError('Specify a cctype!')
    modulename, classname = cctype_class_lookup[cctype]
    mod = importlib.import_module(modulename)
    return getattr(mod, classname)

def valid_cctype(dist):
    """Returns True if dist is a valid DistributionGpm."""
    return dist in cctype_class_lookup

def all_cctypes():
    """Returns a list of all known DistributionGpm."""
    return cctype_class_lookup.keys()

def parse_distargs(dists):
    """Parses a list of cctypes, where distargs are in parenthesis.
    >>> Input ['normal','categorical(k=8)','beta_uc'].
    >>> Output ['normal','categorical','beta_uc'], [None, {'k':8}, None].
    """
    cctypes, distargs = [], []
    for cctype in dists:
        keywords = re.search('\(.*\)', cctype)
        if keywords is not None:
            keywords = keywords.group(0).replace('(','').\
                replace(')','')
            temp = {}
            for subpair in keywords.split(','):
                key, val = subpair.split('=')
                temp[key] = float(val)
            keywords = temp
            cctype = cctype[:cctype.index('(')]
        cctypes.append(cctype)
        distargs.append(keywords)
    return cctypes, distargs
