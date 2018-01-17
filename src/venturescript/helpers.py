# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the 'License');
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an 'AS IS' BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import venture.lite.types as vt
import venture.lite.value as vv

from venture.lite.sp_help import deterministic_typed

from venture.exception import VentureException

# XXX Mutators generally do not exist in Venture.
def dict_set(d, k, v):
    assert isinstance(d, vv.VentureDict)
    d.dict[k] = v

def dict_pop(d, k):
    assert isinstance(d, vv.VentureDict)
    return d.dict.pop(k)

def dict_pop2(d, k):
    assert isinstance(d, vv.VentureDict)
    return dict_pop(d, k) if k in d.dict else vv.VentureNil()

def check_unbound_sp(ripl, name):
    try:
        ripl.sample(name)
        return False
    except VentureException:
        return True

def __venture_start__(ripl):
    if check_unbound_sp(ripl, 'dict_set'):
        ripl.bind_foreign_sp('dict_set', deterministic_typed(
            dict_set,
            [vt.HomogeneousMappingType(vt.AnyType('k'), vt.AnyType('v')),
                vt.AnyType('k'), vt.AnyType('v')],
            vt.NilType()))
    if check_unbound_sp(ripl, 'dict_pop'):
        ripl.bind_foreign_sp('dict_pop', deterministic_typed(
            dict_pop,
            [vt.HomogeneousMappingType(vt.AnyType('k'), vt.AnyType('v')),
                vt.AnyType('k')],
            vt.AnyType('v')))
    if check_unbound_sp(ripl, 'dict_pop2'):
        ripl.bind_foreign_sp('dict_pop2', deterministic_typed(
            dict_pop2,
            [vt.HomogeneousMappingType(vt.AnyType('k'), vt.AnyType('v')),
                vt.AnyType('k')],
            vt.AnyType('v')))
