# -*- coding: utf-8 -*-

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

from gpmcc.utils import test as tu
from particle_engine import ParticleEngine
from particle_dim import ParticleDim
import numpy as np
import matplotlib.pyplot as plt

def logmeanexp(values):
    from scipy.misc import logsumexp
    return logsumexp(values) - np.log(len(values))

# Fix the dataset.
np.random.seed(100)
particles = 20

# set up the data generation
n_rows = 10
view_weights = np.ones(1)
cluster_weights = [ np.array([.5, .5]) ]
cctypes = ['beta_uc']
separation = [.8]
distargs = [None]

# T = np.random.normal(loc=1000, scale=10, size=100)


T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights, cluster_weights,
    cctypes, distargs, separation, return_dims=True)
dim = ParticleDim('normal')
# dim.particle_learn(T[0])

plt.ion(); plt.show()
_, ax = plt.subplots()
def on_click(event):
    # get the x and y coords, flip y from top to bottom
    if event.button == 1:
        if event.inaxes is not None:
            print('data coords %f %f' % (event.xdata, event.ydata))
            dim.particle_learn([event.xdata])
            ax.clear()
            while True:
                dim.gibbs_transition()
                ax.clear()
                dim.plot_dist(ax=ax, Y=np.linspace(0,1,200))
                ax.grid()
                plt.draw()
                plt.pause(0.5)
                print dim.Nobs


plt.connect('button_press_event', on_click)

# engine = ParticleEngine('beta_uc', particles=particles)
# engine.particle_learn(T[0])
# weights = [engine.get_dim(i).weight for i in xrange(particles)]
# dims = [engine.get_dim(i) for i in xrange(particles)]
