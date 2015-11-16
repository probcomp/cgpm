import math
from math import sin
from math import cos
from math import log

import numpy as np
import scipy
import pylab

import gpmcc.utils.general as gu
import gpmcc.utils.sampling as su

TWOPI = 2 * math.pi
LOG2PI = log(2 * math.pi)

def estimate_kappa(N, ssx, scx):
    if N == 0:
        kappa = 10**-6
    elif N == 1:
        kappa = 2*math.pi
    else:
        rbar2 = (ssx/N)**2. + (scx/N)**2.
        rbar = rbar2**.5
        kappa = rbar*(2.-rbar2)/(1.-rbar2)
    return kappa


class VonmisesUC(object):
    """Von Mises data type.
    All data should be in the range [0, 2*pi]
    Does not require additional argumets (distargs=None).
    """

    cctype = 'vonmises_uc'

    def __init__(self, N=0, sum_sin_x=0, sum_cos_x=0, k=1.5, a=1, b=math.pi,
            shape=1, scale=1, distargs=None):
        """
        Keyword arguments:
        N -- numbe of data points assigned to this category. N >= 0
        sum_sin_x -- sufficient statistic sum(sin(X))
        sum_cos_x -- sufficient statistic sum(cos(X))
        k -- vonmises concentration parameter, k > 0
        a -- prior concentration parameter on Von Mises mean, mu. a > 0
        b -- prior mean parameter on Von Mises mean. b is in [0, 2*pi]
        shape -- gamma shape parameter
        scale -- gamma scale parameter
        distargs -- not used
        """
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k > 0
        assert shape > 0
        assert scale > 0

        self.N = N
        self.sum_sin_x = sum_sin_x
        self.sum_cos_x = sum_cos_x

        self.k = k

        self.shape = shape
        self.scale = scale
        self.a = a
        self.b = b

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        assert 0 <= hypers['b'] and hypers['b'] <= TWOPI
        assert hypers['scale'] > 0
        assert hypers['shape'] > 0

        self.a = hypers['a']
        self.b = hypers['b']
        self.scale = hypers['scale']
        self.shape = hypers['shape']

    def set_params(self, params):
        assert params['k'] > 0
        self.k = params['k']

    def insert_element(self, x):
        assert 0 <= x and x <= 2*math.pi
        self.N += 1.0
        self.sum_sin_x += sin(x)
        self.sum_cos_x += cos(x)

    def remove_element(self, x):
        assert 0 <= x and x <= 2*math.pi
        self.N -= 1.0
        if self.N <= 0:
            self.sum_sin_x = 0
            self.sum_cos_x = 0
        else:
            self.sum_sin_x -= sin(x)
            self.sum_cos_x -= cos(x)

    def predictive_logp(self, x):
        assert 0 <= x and x <= 2*math.pi
        return self.calc_predictive_logp(x, self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)

    def marginal_logp(self):
        return self.calc_marginal_logp(self.N, self.sum_sin_x, self.sum_cos_x,
            self.k, self.a, self.b, self.scale, self.shape)

    def update_component_parameters(self):
        n_samples = 25

        D = (0.0, float('Inf'))

        lambda_k = lambda k : self.calc_marginal_logp(self.N, self.sum_sin_x,
            self.sum_cos_x, k, self.a, self.b, self.scale, self.shape)
        self.k = su.mh_sample(self.k, lambda_k, .5, D, burn=n_samples)

    def predictive_draw(self):
        an, bn = self.posterior_update_parameters(self.N, self.sum_sin_x,
            self.sum_cos_x, self.a, self.b, self.k)

        return np.random.vonmises(bn, self.k)

    @staticmethod
    def singleton_logp(x, hypers):
        assert 0 <= x and x <= 2*math.pi
        shape = hypers['shape']
        scale = hypers['scale']
        k = np.random.gamma(shape, scale=scale)

        a = hypers['a']
        b = hypers['b']
        lp = VonmisesUC.calc_predictive_logp(x, 0, 0, 0, a, b, k)

        params = dict(k=k)

        return lp, params

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        ssx = np.sum(np.sin(X))
        scx = np.sum(np.cos(X))
        N = float(len(X))
        k = estimate_kappa(N, ssx, scx)

        grid_interval = TWOPI/n_grid
        grids = dict()
        grids['a'] = gu.log_linspace(1.0/N, TWOPI, n_grid)
        grids['b'] = np.linspace(grid_interval, TWOPI, n_grid)
        grids['shape'] = gu.log_linspace(1.0/N, N, n_grid)
        grids['scale'] = np.linspace(k,k*N*2, n_grid)

        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['a'] = np.random.choice(grids['a'])
        hypers['b'] = np.random.choice(grids['b'])
        hypers['shape'] = np.random.choice(grids['shape'])
        hypers['scale'] = np.random.choice(grids['scale'])

        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, sum_sin_x, sum_cos_x, a, b, k):
        assert 0 <= x and x <= 2*math.pi
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k >= 0

        if x < 0 or x > math.pi*2.0:
            return 0.0

        an, bn = VonmisesUC.posterior_update_parameters(
            N, sum_sin_x, sum_cos_x, a, b, k)

        am, bm = VonmisesUC.posterior_update_parameters(
            N+1.0, sum_sin_x+sin(x), sum_cos_x+cos(x), a, b, k)

        ZN = VonmisesUC.calc_log_Z(an)
        ZM = VonmisesUC.calc_log_Z(am)

        lp = -LOG2PI -  gu.log_bessel_0(k) + ZM - ZN

        return lp

    @staticmethod
    def calc_marginal_logp(N, sum_sin_x, sum_cos_x, k, a, b, scale, shape):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2 * math.pi
        assert k >= 0
        an, bn = VonmisesUC.posterior_update_parameters(
            N, sum_sin_x, sum_cos_x, a, b, k)

        Z0 = VonmisesUC.calc_log_Z(a)
        ZN = VonmisesUC.calc_log_Z(an)

        lp = -(float(N))*(LOG2PI +  gu.log_bessel_0(k)) + ZN-Z0

        return lp + VonmisesUC.calc_k_log_prior(k, scale, shape)

    @staticmethod
    def calc_k_log_prior(k, scale, shape):
        logp = scipy.stats.gamma.logpdf(k, shape, scale=scale)
        return logp

    @staticmethod
    def posterior_update_parameters(N, sum_sin_x, sum_cos_x, a, b, k):
        assert N >= 0
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        assert k >= 0

        p_cos = k*sum_cos_x+a*math.cos(b)
        p_sin = k*sum_sin_x+a*math.sin(b)

        an = (p_cos**2.0+p_sin**2.0)**.5
        bn =  -math.atan2(p_cos,p_sin) + math.pi/2.0;

        return an, bn

    @staticmethod
    def calc_log_Z(a):
        assert a > 0
        return gu.log_bessel_0(a)

    @staticmethod
    def update_hypers(clusters, grids):
        a = clusters[0].a
        b = clusters[0].b
        shape = clusters[0].shape
        scale = clusters[0].scale

        which_hypers = [0,1,2,3]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_a = VonmisesUC.calc_a_conditional_logps(clusters, grids['a'],
                    b, shape, scale)
                a_index = gu.log_pflip(lp_a)
                a = grids['a'][a_index]
            elif hyper == 1:
                lp_b = VonmisesUC.calc_b_conditional_logps(clusters, grids['b'],
                    a, shape, scale)
                b_index = gu.log_pflip(lp_b)
                b = grids['b'][b_index]
            elif hyper == 2:
                lp_scale = VonmisesUC.calc_scale_conditional_logps(clusters,
                    grids['scale'], b, a, shape)
                scale_index = gu.log_pflip(lp_scale)
                scale = grids['scale'][scale_index]
            elif hyper == 3:
                lp_shape = VonmisesUC.calc_shape_conditional_logps(clusters,
                    grids['shape'], a, b, scale)
                shape_index = gu.log_pflip(lp_shape)
                shape = grids['shape'][shape_index]
            else:
                raise ValueError("Invalid hyper.")

        hypers = dict()
        hypers['a'] = a
        hypers['b'] = b
        hypers['shape'] = shape
        hypers['scale'] = scale

        return hypers

    @staticmethod
    def calc_a_conditional_logps(clusters, a_grid, b, scale, shape):
        lps = []
        for a in a_grid:
            lp = VonmisesUC.calc_full_marginal_conditional(clusters, a, b,
                scale, shape)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_b_conditional_logps(clusters, b_grid, a, scale, shape):
        lps = []
        for b in b_grid:
            lp = VonmisesUC.calc_full_marginal_conditional(clusters, a, b,
                scale, shape)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_scale_conditional_logps(clusters, scale_grid, b, a, shape):
        lps = []
        for scale in scale_grid:
            lp = VonmisesUC.calc_full_marginal_conditional(clusters, a, b,
                scale, shape)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_shape_conditional_logps(clusters, shape_grid, a, b, scale):
        lps = []
        for shape in shape_grid:
            lp = VonmisesUC.calc_full_marginal_conditional(clusters, a, b,
                scale, shape)
            lps.append(lp)

        return lps

    @staticmethod
    def calc_full_marginal_conditional(clusters, a, b, scale, shape):
        assert a > 0
        assert 0 <= b and b <= 2*math.pi
        lp = 0
        for cluster in clusters:
            N = cluster.N
            k = cluster.k
            sum_sin_x = cluster.sum_sin_x
            sum_cos_x = cluster.sum_cos_x
            l = VonmisesUC.calc_marginal_logp(N, sum_sin_x, sum_cos_x, k, a, b,
                scale, shape)
            lp += l

        return lp

    @staticmethod
    def calc_full_marginal_conditional_h(clusters, hypers):
        lp = 0
        a = clusters[0].a
        b = clusters[0].b
        scale = clusters[0].scale
        shape = clusters[0].shape
        for cluster in clusters:
            N = cluster.N
            k = cluster.k
            sum_sin_x = cluster.sum_sin_x
            sum_cos_x = cluster.sum_cos_x
            l = VonmisesUC.calc_marginal_logp(N, sum_sin_x, sum_cos_x, k, a, b,
                scale, shape)
            lp += l

        return lp

    @staticmethod
    def plot_dist(X, clusters, distargs=None):
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black"]
        x_min = 0
        x_max = 2*math.pi
        Y = np.linspace(x_min, x_max, 200)
        K = len(clusters)
        pdf = np.zeros((K,200))
        denom = log(float(len(X)))

        a = clusters[0].a
        b = clusters[0].b

        nbins = min([len(X)/5, 50])

        pylab.hist(X, nbins, normed=True, color="black", alpha=.5,
            edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]

        assert math.fabs(sum(np.exp(W)) -1.0) < 10.0**(-10.0)

        for k in range(K):
            vmk = clusters[k].k
            w = W[k]
            N = clusters[k].N
            sum_sin_x = clusters[k].sum_sin_x
            sum_cos_x = clusters[k].sum_cos_x
            for n in range(200):
                y = Y[n]
                pdf[k, n] = np.exp(w + VonmisesUC.calc_predictive_logp(y, N,
                    sum_sin_x, sum_cos_x, a, b, vmk))

            if k >= 8:
                color = "white"
                alpha=.3
            else:
                color = colors[k]
                alpha = .7
            pylab.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)

        pylab.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        pylab.title('vonmises (uncollapsed)')
