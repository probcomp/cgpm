import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
from cgpm.primitives.bernoulli import Bernoulli
from cgpm.utils import general as gu
from cgpm.cgpm import CGpm
from mvdistribution import MultivariateDistributionGpm

# Questions ----------------------------
# 1. What are all the initialization parameters?
# 2.> What does DistributionGpm.incorporate do? Asserts stuff
# 3.> What is there in self.outputs? Variable names.
# 4.> What does DistributionGpm.logpdf do? Asserts stuff
# 5.> What does DistributionGpm.simulate do? Asserts stuff
# 6.> How can I use the Bernoulli CGPM? Just try it out.
# 7. Why doesn't Bernoulli use the params attribute?
# 8. In simulate, why p0 and p1 are computed separately? Normalization?
# ---------------------------------------
def transpose_list(l):
    if l == [] or isinstance(l[0], (float, int)):
        return l
    return map(list, map(None, *l))

def flip(p, size, rng):
    if rng is None:
        rng = gu.gen_rng()
    return rng.choice([0,1], size=size, p=[1-p,p])

def bernoulli_logp(x, p):
    if x not in [0, 1]:
        return -float('inf')
    return np.log((p ** x) * (1-p) ** (1-x))

class IndependentBernoulli(MultivariateDistributionGpm):
    """ Interface for generative population models representing
    multivariate uncollapsed independent Bernoulli distribution.

    x[i] ~ Bernoulli(p[i])
    """

    def __init__(self, outputs=None, params=None, inputs=None,
                 hypers=None, distargs=None, rng=None):
        """
        Parameters
        ----------
        outputs : <int>, optional
            number of variables.
            If not provided, outputs=1
        params : dict{p: list<float>, optional
            If not provided, p = [.5] * number of dimensions
        inputs : Falsy, optional
            Not used
        hypers : Falsy, optional
            Not used
        distargs : Falsy, optional
            Not used
        rng : numpy.random.RandomState, optional
            Source of entropy
         """
        if params is None: params = {}
        MultivariateDistributionGpm.__init__(
            self, outputs, inputs, hypers, params, distargs, rng)
        self.p = params.get('p', [.5] * self.D)
        assert all([prob >= 0 and prob <= 1 for prob in self.p])
        assert len(self.p) == self.D

    def logpdf(self, query, rowid=None, evidence=None):
        """ Type mismatch with cgpm """
        MultivariateDistributionGpm.logpdf(self, rowid, query, evidence)
        if isinstance(query, dict):
            x = query.values()
            rate = self.p[query.keys()]
        elif isinstance(query, list):
            x = query
            rate = self.p
        if any(value not in [0, 1] for value in x):
            return -float('inf')
        return IndependentBernoulli.calc_joint_logp(
            x, rate)

    def simulate(self, query=None, rowid=None, evidence=None, N=None):
        Warning("Irregular output type: <dict> if N=None, else list<dict>")
        if N is not None:
            return [self.simulate(rowid, query, evidence) for i in xrange(N)]
        if query is None:
            x = IndependentBernoulli.sample(p=self.p, rng=self.rng)
            return {self.outputs[i]: x[i] for i in range(self.D)}
        else:
            raise NotImplementedError

    ###################
    # NON-GPM METHODS #
    ###################
    
    def get_params(self):
        return {'p': self.p}

    def simulate_array(self, N=None):
        x = self.simulate(N=N)
        if not isinstance(x, list): x = [x]
        x_list = [[d[i] for i in self.outputs] for d in x]
        return np.array(x_list)

    def plot_samples(self, N=None):
        if N is None: N = 10
        base_p = [self.p]
        x = self.simulate_array(N)

        fig, (ax_1, ax_2) = plt.subplots(
            2, sharex=True, subplot_kw={'adjustable': "box-forced"},
            gridspec_kw={'height_ratios': [1, N]})
        # Ax 1: Bar with probability for each feature
        ax_1.set_yticks(())
        ax_1.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax_1.get_xticklabels(), visible=False)
        ax_1.set_title('$\mathbf{p}$', fontsize=23)
        ax_1.imshow(
            base_p, cmap='Greys', interpolation='none', vmin=0, vmax=1)

        # Ax 2: Binary data
        ax_2.set_xticks(())
        ax_2.set_yticks(())
        ax_2.tick_params(axis=u'both', which=u'both',length=0)
        ax_2.set_ylabel('Entities, N = %s' % (N, ), fontsize=17)
        ax_2.set_xlabel('Features, D = %s' % (self.D, ), fontsize=17)
        ax_2.set_title("$x_i \sim$ Bernoulli($p_i$)", fontsize=17)
        ax_2.imshow(
            x, cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
        fig.tight_layout()
    

        # fig = plt.figure()
        # G = gridspec.GridSpec(8, 1)
        # ax_1 = fig.add_subplot(G[0, :])
        # ax_1.set_yticks(())
        # ax_1.tick_params(axis=u'both', which=u'both', length=0)
        # plt.setp(ax_1.get_xticklabels(), visible=False)
        # ax_1.set_title('$\mathbf{p}$', fontsize=23)
        # ax_1.imshow(
        #     base_p, cmap='Greys', interpolation='none', vmin=0, vmax=1)

        # # Ax 2: Binary data
        # ax_2 = fig.add_subplot(G[1:,:], sharex=ax_1)
        # ax_2.set_xticks(self.outputs, (self.outputs))
        # # ax_2.set_yticks(())
        # ax_2.tick_params(axis=u'both', which=u'both',length=0)
        # ax_2.set_xlabel('$x_i \sim$ Bernoulli($p_i$)', fontsize=17)
        # ax_2.set_ylim(0, x.shape[0])
        # ax_2.imshow(
        #     x, cmap='Greys', interpolation='nearest', vmin=0, vmax=1)

    @staticmethod
    def name():
        return "multivariate independent bernoulli"

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True

    ##################
    # HELPER METHODS #
    ##################
    @staticmethod
    def sample(p, size=None, rng=None):
        """Binary draw from len(p) independent Bernoulli with mean p[i]."""
        assert len(p) > 0
        assert not isinstance(size, list)
        rng_flip = lambda p, size: flip(p, size, rng)
        return transpose_list(map(rng_flip, p, [size] * len(p)))

    @staticmethod
    def calc_joint_logp(x, p):
        marginal_logp = IndependentBernoulli.calc_marg_logp(x, p)
        return sum(marginal_logp)

    @staticmethod
    def calc_marg_logp(x, p):
        assert len(x) == len(p)
        return map(bernoulli_logp, x, p)
        
