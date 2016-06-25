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
import pytest

import numpy as np

from cgpm.venturescript.vscgpm import VsCGpm


source = """
[define make_cgpm (lambda ()
  (do
    ; Population variables.
    (assume sigma (gamma 1 1))

    ; Latent variables.
    (assume x
      (mem (lambda (rowid)
        (normal 0 sigma))))

    ; Output variables.
    (assume simulate_m
      (mem (lambda (rowid w)
        (normal (x rowid) sigma))))

    (assume simulate_y
      (mem (lambda (rowid w)
        (uniform_continuous (- w 10) (+ w 10)))))

    (assume simulators (list simulate_m
                             simulate_y))))]

[define observe_m
  (lambda (rowid w value label)
    (observe (simulate_m ,rowid ,w)
             (atom value) ,label))]

[define observe_y
  (lambda (rowid w value label)
    (observe (simulate_y ,rowid ,w)
             (atom value) ,label))]

[define observers (list observe_m
                        observe_y)]

[define inputs (list 'w)]
"""


def test_wrong_outputs():
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1], inputs=[2], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2,3], inputs=[2], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2,2], inputs=[2], source=source)

def test_wrong_inputs():
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[1], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[], source=source)
    with pytest.raises(ValueError):
        VsCGpm(outputs=[1,2], inputs=[3,4], source=source)
