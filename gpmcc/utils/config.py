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

from gpmcc.cc_types import normal_uc
from gpmcc.cc_types import beta_uc
from gpmcc.cc_types import normal
from gpmcc.cc_types import binomial
from gpmcc.cc_types import multinomial
from gpmcc.cc_types import lognormal
from gpmcc.cc_types import poisson
from gpmcc.cc_types import exponential
from gpmcc.cc_types import exponential_uc
from gpmcc.cc_types import geometric
from gpmcc.cc_types import vonmises
from gpmcc.cc_types import vonmises_uc

dist_class_lookup = {
    'normal'      : normal.Normal,
    'normal_uc'   : normal_uc.NormalUC,
    'beta_uc'     : beta_uc.BetaUC,
    'binomial'    : binomial.Binomial,
    'multinomial' : multinomial.Multinomial,
    'lognormal'   : lognormal.Lognormal,
    'poisson'     : poisson.Poisson,
    'exponential' : exponential.Exponential,
    'exponential_uc' : exponential_uc.ExponentialUC,
    'geometric'   : geometric.Geometric,
    'vonmises'    : vonmises.Vonmises,
    'vonmises_uc' : vonmises_uc.VonmisesUC,
}

dist_collapsed_lookup = {
        'normal'      : False,
        'normal_uc'   : True,
        'beta_uc'     : True,
        'binomial'    : False,
        'multinomial' : False,
        'lognormal'   : False,
        'poisson'     : False,
        'exponential' : False,
        'exponential_uc' : True,
        'geometric'   : False,
        'vonmises'    : False,
        'vonmises_uc' : True,
}

def colors():
    """Returns a list of colors for plotting."""
    return ["red", "blue", "green", "yellow", "orange", "purple", "brown",
        "black"]

def is_uncollapsed(dist):
    """Returns a dict of collapsed, uncollapsed column types."""
    return dist_collapsed_lookup[dist]

def dist_class(dist):
    """Return a dict of class objects for initializing distributions."""
    return dist_class_lookup[dist]

def valid_dist(dist):
    """Returns Ture if dist is a valid distribution."""
    return dist in dist_class_lookup

def all_dists():
    """Returns Ture if dist is a valid distribution."""
    return dist_class_lookup.keys()
