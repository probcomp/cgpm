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

import os
import re

from datetime import datetime

import importlib


cctype_class_lookup = {
    'bernoulli'         : ('cgpm.primitives.bernoulli', 'Bernoulli'),
    'beta'              : ('cgpm.primitives.beta', 'Beta'),
    'categorical'       : ('cgpm.primitives.categorical', 'Categorical'),
    'crp'               : ('cgpm.primitives.crp', 'Crp'),
    'exponential'       : ('cgpm.primitives.exponential', 'Exponential'),
    'geometric'         : ('cgpm.primitives.geometric', 'Geometric'),
    'linear_regression' : ('cgpm.regressions.linreg', 'LinearRegression'),
    'lognormal'         : ('cgpm.primitives.lognormal', 'Lognormal'),
    'normal'            : ('cgpm.primitives.normal', 'Normal'),
    'normal_trunc'      : ('cgpm.primitives.normal_trunc', 'NormalTrunc'),
    'poisson'           : ('cgpm.primitives.poisson', 'Poisson'),
    'random_forest'     : ('cgpm.regressions.forest', 'RandomForest'),
    'vonmises'          : ('cgpm.primitives.vonmises', 'Vonmises'),
}

# https://github.com/posterior/loom/blob/master/doc/using.md#input-format
cctype_loom_lookup = {
    'bernoulli'         : 'boolean',
    'beta'              : 'real',
    'categorical'       : 'categorical',
    'crp'               : 'unbounded_categorical',
    'exponential'       : 'real',
    'geometric'         : 'real',
    'lognormal'         : 'real',
    'normal'            : 'real',
    'normal_trunc'      : 'real',
    'poisson'           : 'count',
    'vonmises'          : 'real',
}

def timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def colors():
    """Returns a list of colors."""
    return ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown',
        'black', 'pink']

def cctype_class(cctype):
    """Return class object for initializing a named GPM (default normal)."""
    if not cctype:
        raise ValueError('Specify a cctype!')
    modulename, classname = cctype_class_lookup[cctype]
    mod = importlib.import_module(modulename)
    return getattr(mod, classname)

def loom_stattype(cctype, distargs):
    # XXX Loom categorical is only up to 256 values; otherwise we need
    # unbounded_categorical (aka crp).
    if cctype == 'categorical' and distargs['k'] > 256:
        cctype = 'crp'
    try:
        return cctype_loom_lookup[cctype]
    except KeyError:
        raise ValueError(
            'Cannot convert cgpm type to loom type: %s' % (cctype,))

def valid_cctype(dist):
    """Returns True if dist is a valid DistributionGpm."""
    return dist in cctype_class_lookup

def all_cctypes():
    """Returns a list of all known DistributionGpm."""
    return list(cctype_class_lookup.keys())

def parse_distargs(dists):
    """Parses a list of cctypes, where distargs are in parenthesis.
    >>> Input ['normal','categorical(k=8)','beta'].
    >>> Output ['normal','categorical','beta'], [None, {'k':8}, None].
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

def check_env_debug():
    debug = os.environ.get('GPMCCDEBUG', None)
    return False if debug is None else int(debug)
