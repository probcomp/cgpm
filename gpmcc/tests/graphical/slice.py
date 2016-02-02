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
import numpy as np
from scipy.integrate import trapz

import gpmcc.utils.general as gu
import gpmcc.utils.sampling as su
import matplotlib.pyplot as plt

def main(num_samples, burn, lag, w):
    alpha = 1.0
    N = 25
    Z = gu.simulate_crp(N, alpha)
    K = max(Z) + 1

    # CRP with gamma prior.
    log_pdf_fun = lambda alpha : gu.logp_crp_unorm(N, K, alpha) - alpha
    proposal_fun = lambda : np.random.gamma(1.0, 1.0)
    D = (0, float('Inf'))

    samples = su.slice_sample(proposal_fun, log_pdf_fun, D,
        num_samples=num_samples, burn=burn, lag=lag, w=w)

    minval = min(samples)
    maxval = max(samples)
    xvals = np.linspace(minval, maxval, 100)
    yvals = np.array([math.exp(log_pdf_fun(x)) for x in xvals])
    yvals /= trapz(xvals, yvals)

    fig, ax = plt.subplots(2,1)

    ax[0].hist(samples, 50, normed=True)

    ax[1].hist(samples, 100, normed=True)
    ax[1].plot(xvals,-yvals, c='red', lw=3, alpha=.8)
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--burn', default=10, type=int)
    parser.add_argument('--lag', default=5, type=int)
    parser.add_argument('--w', default=1.0, type=float)

    args = parser.parse_args()

    num_samples = args.num_samples
    burn = args.burn
    lag = args.lag
    w = args.w

    main(num_samples, burn, lag, w)
