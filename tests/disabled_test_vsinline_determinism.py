# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
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

import matplotlib.pyplot as plt
import numpy as np

from cgpm.utils import general as gu
from cgpm.venturescript.vsinline import InlineVsCGpm

# The CGPM for X with seed 4.
vsx_s4 = InlineVsCGpm([0], [],
    expression='(lambda () (uniform_continuous -4.71 4.71))',
    rng=gu.gen_rng(4))

# The CGPM for Y with seed 4.
vsy_s4 = InlineVsCGpm([1], [0],
    expression='''(lambda (x) (if (> (cos x) 0) (uniform_continuous (- (cos x) .5) (cos x)) (uniform_continuous (cos x) (+ (cos x) .5))))''',
    rng=gu.gen_rng(4))

# The CGPM for Y with seed 5.
vsy_s5 = InlineVsCGpm([1], [0],
    expression='''(lambda (x) (if (> (cos x) 0) (uniform_continuous (- (cos x) .5) (cos x)) (uniform_continuous (cos x) (+ (cos x) .5))))''',
    rng=gu.gen_rng(5))

# Simulate the uniform x from vsx.
samples_x4 = vsx_s4.simulate(0, [0], N=200)

# Simulate Y from each of the seed 4 and 5.
samples_y4 = [vsy_s4.simulate(0, [1], sx) for sx in samples_x4]
samples_y5 = [vsy_s5.simulate(0, [1], sx) for sx in samples_x4]

# Convert all samples from dictionaries to lists.
xs4 = [s[0] for s in samples_x4]
ys4 = [s[1] for s in samples_y4]
ys5 = [s[1] for s in samples_y5]

# Compute the noise at each data point for both sample sets.
errors1 = np.cos(xs4)-ys4
errors2 = np.cos(xs4)-ys5

# Plot the joint query.
fig, ax = plt.subplots()
ax.scatter(xs4, ys4, color='blue', alpha=.4) # There is no noise in the cosx.
ax.scatter(xs4, ys5, color='red', alpha=.4)  # This is noise in the cosx.
ax.set_xlim([-1.5*np.pi, 1.5*np.pi])
ax.set_ylim([-1.75, 1.75])
for x in xs4:
    ax.vlines(x, -1.75, -1.65, linewidth=.5)
ax.grid()

# Plot the errors.
fig, ax = plt.subplots()
ax.scatter(xs4, errors1, color='blue', alpha=.4)
ax.scatter(xs4, errors2, color='red', alpha=.4)
ax.set_xlabel('value of x')
ax.set_ylabel('error of cos(x) - y')
ax.grid()
