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

import math

import numpy as np

from matplotlib import pyplot as plt
from scipy.integrate import trapz

from cgpm.utils import general as gu
from cgpm.utils import sampling as su


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
