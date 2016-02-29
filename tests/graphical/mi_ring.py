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

import numpy as np
import matplotlib.pyplot as plt

from gpmcc import state
from gpmcc.utils import inference as iu
from gpmcc.utils import test as tu

W = [.9, .75, .5, .25, .1]
n_data_sets = 3
n_samples = 5

N = 250
i = 0

for kernel in xrange(2):
    MI = np.zeros((n_data_sets*n_samples, len(W)))
    c = 0
    for w in W:
        r = 0
        for ds in xrange(n_data_sets):
            np.random.seed(r+ds)
            X = np.asarray(tu.gen_ring(N, width=w)).T + 100
            print X
            for _ in range(n_samples):
                S = state.State(X, ['normal','normal'], distargs=[None, None])
                S.transition(N=200)
                mi = iu.mutual_information(S, 0, 1)
                MI[r,c] = mi
                print 'w: %1.2f, MI: %1.6f' % (w, mi)
                print '%i of %i' % (i+1, len(W)*n_data_sets*n_samples*2)
                del S
                i += 1
                r += 1
        c += 1
    w_labs = [str(w) for w in W]
    ax = plt.subplot(1,2,kernel+1)
    plt.boxplot(MI)
    plt.ylim([0,1])
    plt.ylabel('MI')
    plt.xlabel('Ring Width')
    plt.title('Kernel %i' % kernel)
    ax.set_xticklabels(w_labs)

plt.show()
