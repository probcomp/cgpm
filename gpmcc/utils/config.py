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

import re

from gpmcc.dists import normal_uc
from gpmcc.dists import beta_uc
from gpmcc.dists import normal
from gpmcc.dists import bernoulli
from gpmcc.dists import categorical
from gpmcc.dists import lognormal
from gpmcc.dists import poisson
from gpmcc.dists import exponential
from gpmcc.dists import exponential_uc
from gpmcc.dists import geometric
from gpmcc.dists import vonmises

distgpm_class_lookup = {
    'normal'            : normal.Normal,
    'normal_uc'         : normal_uc.NormalUC,
    'beta_uc'           : beta_uc.BetaUC,
    'bernoulli'         : bernoulli.Bernoulli,
    'categorical'       : categorical.Categorical,
    'lognormal'         : lognormal.Lognormal,
    'poisson'           : poisson.Poisson,
    'exponential'       : exponential.Exponential,
    'exponential_uc'    : exponential_uc.ExponentialUC,
    'geometric'         : geometric.Geometric,
    'vonmises'          : vonmises.Vonmises,
}

def colors():
    """Returns a list of colors."""
    return ["red", "blue", "green", "yellow", "orange", "purple", "brown",
        "black"]

def distgpm_class(dist):
    """Return a class object for initializing a named DistributionGpm."""
    return distgpm_class_lookup[dist]

def valid_distgpm(dist):
    """Returns True if dist is a valid DistributionGpm."""
    return dist in distgpm_class_lookup

def all_distgpms():
    """Returns a list of all known DistributionGpm."""
    return distgpm_class_lookup.keys()

def parse_distargs(dists):
    """Parses a list of distgpms, where distargs are in parenthesis.
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
