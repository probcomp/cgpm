import gpmcc.utils.sampling as su
import gpmcc.utils.general as utils
from scipy.stats import gamma
from numpy.random import gamma as gamrnd
from scipy.integrate import simps, trapz
import random
import math
import numpy
import pylab

def main(num_samples, burn, lag, w):

    alpha = 1.0
    N = 25
    Z, Nk, K = utils.crp_gen(N, alpha)

    # crp with gamma prior
    log_prior_fun = lambda a: -a
    log_pdf_lambda = lambda a : utils.unorm_lcrp_post(a, N, K, log_prior_fun)
    proposal_fun = lambda : gamrnd(1.0,1.0)
    D = (0, float('Inf'))

    samples = su.slice_sample(proposal_fun, log_pdf_lambda, D, num_samples=num_samples, burn=burn, lag=lag, w=w)

    minval = min(samples)
    maxval = max(samples)
    xvals = numpy.linspace(minval, maxval, 100)

    yvals = numpy.array([ math.exp(log_pdf_lambda(x)) for x in xvals])
    yvals /= trapz(xvals, yvals)

    ax = pylab.subplot(2,1,1)
    pylab.hist(X, 50, normed=True)

    ax_1=pylab.subplot(2,1,2)
    pylab.hist(samples,100,normed=True)
    pylab.plot(xvals,-yvals,c='red',lw=3, alpha=.8)
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
    




