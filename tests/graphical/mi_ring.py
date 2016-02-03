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
