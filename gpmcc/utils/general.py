import numpy
import random
from math import log
from scipy.special import i0 as bessel_0
from scipy.misc import logsumexp
from scipy.special import gammaln
import math
import csv

def log_bessel_0(x):
    besa = bessel_0(x)
    # if bessel_0(a) is inf, then use the eponential approximation to 
    # prevent numericala overflow
    if math.isinf(besa):
        I0 = x - .5*log(2*math.pi*x)
    else:
        I0 = log(besa)

    return I0

def log_normalize(logp):
    """
    Normalizes a numpy array of log probabilites
    """
    return logp - logsumexp(logp)

def lcrp(N, Nk, alpha):
    """
    Returns the log probability under crp of the count vector Nk given
    concentration parameter alpha. N is the total number of entries
    """
    N = float(N)
    k = float(len(Nk)) # number of classes
    l = numpy.sum(gammaln(Nk))+k*log(alpha)+gammaln(alpha)-gammaln(N+alpha); 
    return l

def unorm_lcrp_post(alpha, N, K, log_prior_fun):
    """
    returns the log of unnormalized P(alpha|N,K)
    Arguments:
    -- alpha: crp alpha parameter. float greater than 0
    -- N: number of point in partition, Z
    -- K: number of partitions in Z
    -- log_prior_fun: function of alpha. log P(alpha)
    """
    return gammaln(alpha)+float(K)*log(alpha)-gammaln(alpha+float(N))+log_prior_fun(alpha)

def log_pflip(P):
    """
    Multinomial draw from a vector P of log probabilities
    """
    if len(P) == 1:
        return 0

    Z = logsumexp(P)
    P -= Z

    NP = numpy.exp(numpy.copy(P))

    assert( math.fabs(1.0-sum(NP)) < 10.0**(-10.0) )

    return pflip(NP)

def pflip(P):
    """
    Multinomial draw from a vector P of probabilities
    """
    if len(P) == 1:
        return 0

    P /= sum(P)

    if not (math.fabs(1.0-sum(P)) < 10.0**(-10.0)):
        pdb.set_trace()

    p_minus = 0
    r = random.random()
    for i in range(len(P)):
        P[i] += p_minus
        p_minus = P[i]
        if r < p_minus:
            return i

    # return numpy.choice(range(len(P)),1,p=P)[0]

    raise IndexError("pflip:failed to find index")

def log_linspace(a, b, n):
    """
    linspace from a to b with n entries over log scale (mor entries at smaller values)
    """
    return numpy.exp(numpy.linspace(log(a), log(b), n))

def log_nchoosek(n, k):
    """
    log(nchoosek(n,k)) with overflow protection
    """
    assert n >= 0
    assert k >= 0
    assert n >= k
    if n == 0 or k == 0 or n == k:
        return 0

    return log(n)+gammaln(n) - log(k) - gammaln(k) - log(n-k) - gammaln(n-k)

def line_quad(x,y):
    """
    quadrature over x, where y = f(x). Uses triangle rule.
    """
    s = 0
    for i in range(1,len(x)):
        a = x[i-1]
        b = x[i]
        fa = y[i-1]
        fb = y[i]

        s += (b-a)*(fa+fb)/2

    return s


def crp_gen(N, alpha):
    """
    Generates a random, N-length partition from the CRP with parameter alpha
    """
    assert(N > 0)
    assert(alpha > 0.0)
    alpha = float(alpha)

    partition = numpy.zeros(N,dtype=int);
    Nk = [1]
    for i in range(1,N):
        K = len(Nk);
        ps = numpy.zeros(K+1)
        for k in range(K):
            # get the number of people sitting at table k
            ps[k] = float(Nk[k])
        
        ps[K] = alpha

        ps /= (float(i)-1+alpha)

        assignment = pflip(ps)

        if assignment == K:
            Nk.append(1)
        elif assignment < K:
            Nk[assignment] += 1
        else:
            raise ValueError("invalid assignment: %i, max=%i" % (assignment, K))

        partition[i] = assignment

    assert max(partition)+1 == len(Nk)
    assert len(partition)==N
    assert sum(Nk) == N

    K = len(Nk)

    if K > 1:
        random.shuffle(partition)

    return numpy.array(partition), Nk, K

def kl_array(support, log_true, log_inferred, is_discrete):
    """
    Inputs:
    -- support: numpy array of support intervals
    -- log_true: log pdf at support for the "true" distribution
    -- log_inferred: log pdf at support for the distribution to test against the
    "true" distribution
    -- is_discrete: is this a discrete variable True/False
    Returns:
    - KL divergence
    """

    # KL divergence formula, recall X and Y are log
    F = (log_true-log_inferred)*numpy.exp(log_true)
    if is_discrete:
        kld = numpy.sum(F)
    else:
        # trapezoidal quadrature
        intervals = numpy.diff(support)
        fs = F[:-1] + (numpy.diff(F) / 2.0)
        kld = numpy.sum(intervals*fs)

    return kld

def bincount(X, bins=None):
    """
    Returns the frequency of each entry in bins of X. If bins is not epecified,
    then bins is [0,1,..,max(X)].
    """

    Y = numpy.array(X, dtype=int)
    if bins == None:
        minval = numpy.min(Y)
        maxval = numpy.max(Y)

        bins = range(minval, maxval+1)

    # if not isinstance(bins, list):
    #     raise TypeError('bins should be a list')

    counts = [0]*len(bins)

    for y in Y:
        bin_index = bins.index(y)
        counts[bin_index] += 1

    assert len(counts) == len(bins)
    assert sum(counts) == len(Y)

    return counts

def csv_to_list(filename):
    """reads the csv filename into a list of lists"""
    T = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            T.append(row)

    return T

def csv_to_data_and_colnames(filename):
    TC = csv_to_list(filename)
    colnames = list(TC[0])
    Y = numpy.genfromtxt(filename, delimiter=',', skip_header=1)

    if len(Y.shape) == 1:
        return [Y], colnames 
    else:
        X = []
        for col in range(Y.shape[1]):
            X.append(Y[:,col])

        return X, colnames

def clean_data(X, cctypes):
    """
    Makes sure that descrete data columns are integer types
    """
    is_discrete = {
    'normal'      : False,
    'normal_uc'   : False,
    'binomial'    : False,  # should be 0.0 or 1.0
    'multinomial' : True,
    'lognormal'   : False,
    'poisson'     : False,
    'vonmises'    : False,
    'vonmises_uc' : False,
    } 

    for i in range(len(X)):
        if is_discrete[cctypes[i]]:
            X[i] = numpy.array(X[i], dtype=int)

    return X


