import math
from math import log

import numpy as np
import pylab
from scipy.special import gammaln

import gpmcc.utils.general as gu

LOG2 = log(2.0)
LOGPI = log(math.pi)
LOG2PI = log(2*math.pi)

class Normal(object):
    """Normal data with normal prior on mean and gamma prior on precision.
    Does not require additional argumets (distargs=None).
    """

    cctype = 'normal'

    def __init__(self, N=0, sum_x=0, sum_x_sq=0, m=0, r=1, s=1, nu=1,
            distargs=None):
        """
        Optional arguments:
        -- N: number of data points
        -- sum_x: suffstat, sum(X)
        -- sum_x_sq: suffstat, sum(X^2)
        -- m: hyperparameter
        -- r: hyperparameter
        -- s: hyperparameter
        -- nu: hyperparameter
        -- distargs: not used
        """
        assert s > 0.0
        assert r > 0.0
        assert nu > 0.0

        self.N = N
        self.sum_x = sum_x
        self.sum_x_sq = sum_x_sq
        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

    def set_hypers(self, hypers):
        assert hypers['s'] > 0.0
        assert hypers['r'] > 0.0
        assert hypers['nu'] > 0.0

        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def insert_element(self, x):
        if math.isnan(x):
            return
        self.N += 1.0
        self.sum_x += x
        self.sum_x_sq += x*x

    def remove_element(self, x):
        if math.isnan(x):
            return 
        self.N -= 1.0
        if self.N == 0:
            self.sum_x = 0.0
            self.sum_x_sq = 0.0
        else:
            self.sum_x -= x
            self.sum_x_sq -= x*x


    def predictive_logp(self, x):
        return self.calc_predictive_logp(x, self.N, self.sum_x, self.sum_x_sq,
                                        self.m, self.r, self.s, self.nu)

    def singleton_logp(self, x):
        # return self.calc_predictive_logp(x, 0, 0, 0,self.m, self.r, self.s, self.nu)
        return self.calc_marginal_logp(1, x, x*x, self.m, self.r, self.s,
            self.nu)

    def marginal_logp(self):
        return self.calc_marginal_logp(self.N, self.sum_x, self.sum_x_sq,
            self.m, self.r, self.s, self.nu)

    def predictive_draw(self):
        rn, nun, mn, sn = Normal.posterior_update_parameters(
            self.N, self.sum_x, self.sum_x_sq, self.m, self.r, self.s, self.nu)
        coeff = ( ((sn/2.0)*(rn+1.0)) / ((nun/2.0)*rn) )**.5
        draw = np.random.standard_t(nun) * coeff + mn
        return draw

    @staticmethod
    def construct_hyper_grids(X,n_grid=30):
        grids = dict()
        ssqdev = np.var(X) * float(len(X))

        grids['r'] = gu.log_linspace(1.0/float(len(X)),float(len(X)), n_grid)
        grids['s'] = gu.log_linspace(ssqdev/100.0, ssqdev, n_grid)
        grids['nu'] = gu.log_linspace(1.0,float(len(X)), n_grid) # df >= 1
        grids['m'] = np.linspace(min(X),max(X), n_grid)

        grids['m'] = np.linspace(-2.0,2.0, n_grid) # for gewek
        grids['s'] = gu.log_linspace(.01, 3.0, n_grid) # for geweke

        return grids

    @staticmethod
    def init_hypers(grids, X=None):
        hypers = dict()
        hypers['m'] = np.random.choice(grids['m'])
        hypers['s'] = np.random.choice(grids['s'])
        hypers['r'] = np.random.choice(grids['r'])
        hypers['nu'] = np.random.choice(grids['nu'])

        return hypers

    @staticmethod
    def calc_predictive_logp(x, N, sum_x, sum_x_sq, m, r, s, nu):
        if math.isnan(x):
            return 0

        rn, nun, mn, sn = Normal.posterior_update_parameters(N, sum_x, sum_x_sq,
            m, r, s, nu)

        rm, num, mm, sm = Normal.posterior_update_parameters(
            N+1, sum_x+x, sum_x_sq+x*x, m, r, s, nu)

        ZN = Normal.calc_log_Z(rn, sn, nun)
        ZM = Normal.calc_log_Z(rm, sm, num)

        return -.5*LOG2PI + ZM - ZN

    @staticmethod
    def calc_marginal_logp(N, sum_x, sum_x_sq, m, r, s, nu):
        rn, nun, mn, sn = Normal.posterior_update_parameters(N, sum_x, sum_x_sq,
            m, r, s, nu)

        Z0 = Normal.calc_log_Z(r, s, nu)
        ZN = Normal.calc_log_Z(rn, sn, nun)

        return - (float(N) / 2.0) * LOG2PI + ZN - Z0

    @staticmethod
    def update_hypers(clusters, grids):
        # resample hypers
        m = clusters[0].m
        s = clusters[0].s
        r = clusters[0].r
        nu = clusters[0].nu

        which_hypers = [0,1,2,3]
        np.random.shuffle(which_hypers)

        for hyper in which_hypers:
            if hyper == 0:
                lp_m = Normal.calc_m_conditional_logps(clusters, grids['m'], r,
                    s, nu)
                m_index = gu.log_pflip(lp_m)
                m = grids['m'][m_index]
            elif hyper == 1:
                lp_s = Normal.calc_s_conditional_logps(clusters, grids['s'], m,
                    r, nu)
                s_index = gu.log_pflip(lp_s)
                s = grids['s'][s_index]
            elif hyper == 2:
                lp_r = Normal.calc_r_conditional_logps(clusters, grids['r'], m,
                    s, nu)
                r_index = gu.log_pflip(lp_r)
                r = grids['r'][r_index]
            elif hyper == 3:
                lp_nu = Normal.calc_nu_conditional_logps(clusters, grids['nu'],
                    m, r, s)
                nu_index = gu.log_pflip(lp_nu)
                nu = grids['nu'][nu_index]
            else:
                raise ValueError("Invalid hyper.")

        hypers = dict()
        hypers['m'] = m
        hypers['s'] = s
        hypers['r'] = r
        hypers['nu'] = nu

        return hypers

    @staticmethod
    def posterior_update_parameters(N, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + float(N)
        nun = nu + float(N)
        mn = (r*m + sum_x)/rn
        sn = s + sum_x_sq + r*m*m - rn*mn*mn

        if sn == 0:
            print("posterior_update_parameters: sn(0) truncated")
            sn = s

        return rn, nun, mn, sn

    @staticmethod
    def calc_log_Z(r, s, nu):
        Z = ((nu+1.0)/2.0)*LOG2 + .5*LOGPI - .5*log(r) - (nu/2.0)*log(s) + \
            gammaln(nu/2.0)
        return Z

    @staticmethod
    def calc_m_conditional_logps(clusters, m_grid, r, s, nu):
        lps = []
        for m in m_grid:
            lp = Normal.calc_full_marginal_conditional(clusters, m, r, s, nu)
            lps.append(lp)
        return lps

    @staticmethod
    def calc_r_conditional_logps(clusters, r_grid, m, s, nu):
        lps = []
        for r in r_grid:
            lp = Normal.calc_full_marginal_conditional(clusters, m, r, s, nu)
            lps.append(lp)
        return lps

    @staticmethod
    def calc_s_conditional_logps(clusters, s_grid, m, r, nu):
        lps = []
        for s in s_grid:
            lp = Normal.calc_full_marginal_conditional(clusters, m, r, s, nu)
            lps.append(lp)
        return lps

    @staticmethod
    def calc_nu_conditional_logps(clusters, nu_grid, m, r, s):
        lps = []
        for nu in nu_grid:
            lp = Normal.calc_full_marginal_conditional(clusters, m, r, s, nu)
            lps.append(lp)
        return lps

    @staticmethod
    def calc_full_marginal_conditional(clusters, m, r, s, nu):
        # lp = gamma.logpdf(r, 1.0, 1.0) + gamma.logpdf(s, 1.0, 1.0) + gamma.logpdf(nu, 1.0, 1.0)
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            sum_x_sq = cluster.sum_x_sq
            l = Normal.calc_marginal_logp(N, sum_x, sum_x_sq, m, r, s, nu)
            lp += l
        return lp

    @staticmethod
    def calc_full_marginal_conditional_h(clusters, hypers):
        m = clusters[0].m
        r = clusters[0].r
        s = clusters[0].s
        nu = clusters[0].nu
        # lp = gamma.logpdf(r, 1.0, 1.0) + gamma.logpdf(s, 1.0, 1.0) + gamma.logpdf(nu, 1.0, 1.0)
        lp = 0
        for cluster in clusters:
            N = cluster.N
            sum_x = cluster.sum_x
            sum_x_sq = cluster.sum_x_sq
            l = Normal.calc_marginal_logp(N, sum_x, sum_x_sq, m, r, s, nu)
            lp += l
        return lp

    @staticmethod
    def plot_dist(X, clusters, distargs=None):
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black"]
        x_min = min(X)
        x_max = max(X)
        Y = np.linspace(x_min, x_max, 200)
        K = len(clusters)
        pdf = np.zeros((K,200))
        denom = log(float(len(X)))

        m = clusters[0].m
        s = clusters[0].s
        r = clusters[0].r
        nu = clusters[0].nu

        nbins = min([len(X)/5, 50])

        pylab.hist(X, nbins, normed=True, color="black", alpha=.5,
            edgecolor="none")

        W = [log(clusters[k].N) - denom for k in range(K)]

        for k in range(K):
            w = W[k]
            N = clusters[k].N
            sum_x = clusters[k].sum_x
            sum_x_sq = clusters[k].sum_x_sq
            for n in range(200):
                y = Y[n]
                pdf[k, n] = np.exp(w + Normal.calc_predictive_logp(y, N,
                    sum_x, sum_x_sq, m, r, s, nu))

            if k >= 8:
                color = "white"
                alpha=.3
            else:
                color = colors[k]
                alpha=.7
            pylab.plot(Y, pdf[k,:], color=color, linewidth=5, alpha=alpha)

        pylab.plot(Y, np.sum(pdf, axis=0), color='black', linewidth=3)
        pylab.title('normal')
