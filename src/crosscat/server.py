#!/usr/bin/env python
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

"""
Synopsis:
    A simple XML-RPC server.
"""

from SimpleXMLRPCServer import SimpleXMLRPCServer


import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils import general as gu


X = [[1,     np.nan,     2,         -1,         np.nan  ],
     [1,     3,          2,         -1,         -5      ],
     [18,    -7,         -2,        11,         -12     ],
     [1,     np.nan,     np.nan,    np.nan,     np.nan  ],
     [18,    -7,         -2,        11,         -12     ]]

state = State(X, outputs=range(5), cctypes=['normal']*5,
    Zv={0:0, 1:0, 2:0, 3:1, 4:1}, rng=gu.gen_rng(0),)

node = 'localhost'
port = 8000
server = SimpleXMLRPCServer((node, port), allow_none=True)
print "Listening on {} at port {} ...".format(node, port)
server.register_instance(state)
server.serve_forever()
