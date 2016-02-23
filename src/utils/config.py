# -*- coding: utf-8 -*-

# The MIT License (MIT)

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

import re
from datetime import datetime

import importlib

cctype_class_lookup = {
    'normal'            : ('gpmcc.dists.normal', 'Normal'),
    'normal_trunc'      : ('gpmcc.dists.normal_trunc', 'NormalTrunc'),
    'beta_uc'           : ('gpmcc.dists.beta_uc', 'BetaUC'),
    'bernoulli'         : ('gpmcc.dists.bernoulli', 'Bernoulli'),
    'categorical'       : ('gpmcc.dists.categorical', 'Categorical'),
    'lognormal'         : ('gpmcc.dists.lognormal', 'Lognormal'),
    'linreg'            : ('gpmcc.dists.linreg', 'LinearRegression'),
    'poisson'           : ('gpmcc.dists.poisson', 'Poisson'),
    'exponential'       : ('gpmcc.dists.exponential', 'Exponential'),
    'geometric'         : ('gpmcc.dists.geometric', 'Geometric'),
    'vonmises'          : ('gpmcc.dists.vonmises', 'Vonmises'),
}

def timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def colors():
    """Returns a list of colors."""
    return ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown',
        'black', 'pink']

def cctype_class(cctype):
    """Return a class object for initializing a named DistributionGpm."""
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
