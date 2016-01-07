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

import gpmcc.utils.sampling as su
import gpmcc.utils.general as gu
from scipy.integrate import trapz
import math
import numpy as np
import pylab

def main(num_samples, burn, lag, w):
    alpha = 1.0
    N = 25
    Z, Nk, K = gu.crp_gen(N, alpha)

    # crp with gamma prior
    log_prior_fun = lambda a: -a
    log_pdf_lambda = lambda a : gu.unorm_lcrp_post(a, N, K, log_prior_fun)
    proposal_fun = lambda : np.random.gamma(1.0, 1.0)
    D = (0, float('Inf'))

    samples = su.slice_sample(proposal_fun, log_pdf_lambda, D,
        num_samples=num_samples, burn=burn, lag=lag, w=w)

    minval = min(samples)
    maxval = max(samples)
    xvals = np.linspace(minval, maxval, 100)
    yvals = np.array([math.exp(log_pdf_lambda(x)) for x in xvals])
    yvals /= trapz(xvals, yvals)

    ax = pylab.subplot(2,1,1)
    pylab.hist(samples, 50, normed=True)

    ax_1 = pylab.subplot(2,1,2)
    pylab.hist(samples, 100, normed=True)
    pylab.plot(xvals,-yvals, c='red', lw=3, alpha=.8)
    pylab.xlim(ax.get_xlim())
    pylab.ylim(ax.get_ylim())
    pylab.show()

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
