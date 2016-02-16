# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
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

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm

import gpmcc.utils.general as gu
import gpmcc.utils.sampling as su
import gpmcc.utils.plots as pu

# GOOD
# seed=5, start=3, w=.5
# seed=5, start=5, w=1

np.random.seed(5)

logpdf_target = lambda x : np.log(.5*norm.pdf(x,1,1) + .5*norm.pdf(x,5,.75))
x_start = 3
D = (0, float('inf'))

metadata = su.slice_sample(x_start, logpdf_target, D,
    num_samples=20, burn=1, lag=1, w=.5)

samples = metadata['samples']

minval = 0.1
maxval = 10
xvals = np.linspace(minval, maxval, 100)
yvals = np.array([math.exp(logpdf_target(x)) for x in xvals])

fig, ax = plt.subplots()
ax.grid()

plt.ion()
plt.show()
ptime = .5

# pu.plot_samples(samples, h=.025, ax=ax)
ax.plot(xvals, yvals/trapz(yvals, xvals), c='red', lw=3, alpha=.8)
ax.set_xlim([0, 7])
ax.set_ylim([0, ax.get_ylim()[1]])

last_x = x_start
for i in xrange(len(metadata['samples'])):
    u = metadata['u'][i]
    r = metadata['r'][i]
    a_out = metadata['a_out'][i]
    b_out = metadata['b_out'][i]
    x_proposal = metadata['x_proposal'][i]
    sample = metadata['samples'][i]

    # Accumulate artists to delete.
    to_delete = []
    # Plot the uniform proposal (0, f(x))
    plt.pause(ptime)
    to_delete.append(
        ax.vlines(last_x, 0, np.exp(logpdf_target(last_x))/trapz(yvals, xvals),
        color='green'))
    # Plot the auxiliary variable u.
    plt.pause(ptime)
    to_delete.append(
        ax.scatter(last_x, np.exp(u), marker='o', color='b'))
    # Plot growing a.
    for a in a_out:
        plt.pause(ptime)
        to_delete.append(ax.hlines(np.exp(u), a, last_x))
        to_delete.append(ax.vlines(a, np.exp(u), np.exp(u)+.01))
    # Plot growing b.
    for b in b_out:
        plt.pause(ptime)
        to_delete.append(ax.hlines(np.exp(u), last_x, b))
        to_delete.append(ax.vlines(b, np.exp(u), np.exp(u)+.01))
    # # Plot the shrinking.
    # for xp in x_proposal[:-1]:
    #     plt.pause(ptime)
    #     ax.scatter(xp, np.exp(u), marker='o', color='r')
    # Plot the final sample.
    plt.pause(ptime)
    ax.scatter(sample, np.exp(u), marker='o', color='r')
    # Plot a hline at the final sample.
    plt.pause(ptime)
    ax.vlines(sample, 0, 0.025, linewidth=2)
    last_x = sample
    # Remove the uline
    plt.pause(ptime)
    for td in to_delete:
        td.remove()
