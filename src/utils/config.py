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

from gpmcc.dists import beta_uc
from gpmcc.dists import normal
from gpmcc.dists import normal_trunc
from gpmcc.dists import bernoulli
from gpmcc.dists import categorical
from gpmcc.dists import lognormal
from gpmcc.dists import poisson
from gpmcc.dists import exponential
from gpmcc.dists import geometric
from gpmcc.dists import vonmises

cctype_class_lookup = {
    'normal'            : normal.Normal,
    'normal_trunc'      : normal_trunc.NormalTrunc,
    'beta_uc'           : beta_uc.BetaUC,
    'bernoulli'         : bernoulli.Bernoulli,
    'categorical'       : categorical.Categorical,
    'lognormal'         : lognormal.Lognormal,
    'poisson'           : poisson.Poisson,
    'exponential'       : exponential.Exponential,
    'geometric'         : geometric.Geometric,
    'vonmises'          : vonmises.Vonmises,
}

def timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def colors():
    """Returns a list of colors."""
    return ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown',
        'black', 'pink']

def cctype_class(dist):
    """Return a class object for initializing a named DistributionGpm."""
    return cctype_class_lookup[dist]

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
