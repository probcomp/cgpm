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
import pylab

import gpmcc.utils.inference as iu
from gpmcc import state

# rho_list = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
rho_list = [.6, .7, .8, .9, 1.0]
n_data_sets = 3
n_samples = 5
N = 50
mu = np.zeros(2)
i = 0
distargs = [None,None]

# for kernel in range(2):
for kernel in range(2):
    L = np.zeros((n_data_sets*n_samples, len(rho_list)))
    c = 0
    for rho in rho_list:
        r = 0
        for ds in range(n_data_sets):
            # seed control so that data is always the same
            np.random.seed(r+ds)
            sigma = np.array([[1,rho],[rho,1]])
            X = np.random.multivariate_normal(mu,sigma,N)
            for _ in range(n_samples):
                S = state.State(X, ['normal']*2, Zv=[0,0], distargs=distargs)
                S.transition(N=100)
                MI = iu.mutual_information(S, 0, 1)
                linfoot = iu.mutual_information_to_linfoot(MI)
                # del S
                L[r,c] = linfoot
                print \
                    "rho: %1.2f, MI: %1.6f, Linfoot: %1.6f" %(rho, MI, linfoot)
                print \
                    "%i of %i" % (i+1, len(rho_list)*n_data_sets*n_samples*2)
                del S
                i += 1
                r += 1
        c += 1

    rho_labs = [str(rho) for rho in rho_list]
    ax = pylab.subplot(1,2,kernel+1)
    pylab.boxplot(L)
    pylab.ylim([0,1])
    pylab.ylabel('Linfoot')
    pylab.xlabel('rho')
    pylab.title("kernel %i" % kernel)
    ax.set_xticklabels(rho_labs)

pylab.show()
