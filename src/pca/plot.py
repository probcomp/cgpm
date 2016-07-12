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

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from cgpm.pca.ppca import PPCA
from cgpm.utils import general as gu
from cgpm.utils import config as cu


rng = gu.gen_rng(12)

def scatter_classes(x, classes, ax=None):
    if ax is None:
        _fig, ax = plt.subplots()
    ax = plt.gca() if ax is None else ax
    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.Normalize(
        vmin=np.min(classes), vmax=np.max(classes))
    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)
    ax.scatter(x[:,0], x[:,1], color=colors)
    return ax

def fillna(X, p):
    X = np.copy(X)
    a, b = X.shape
    n_entries = a*b
    n_missing = int(a*b*p)
    i_missing_flat = rng.choice(range(n_entries), size=n_missing, replace=False)
    i_missing_cell = np.unravel_index(i_missing_flat, (a,b))
    for i, j in zip(*i_missing_cell):
        X[i,j] = np.nan
    return X

if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    timestamp = cu.timestamp()

    # for p in np.linspace(0, .7, 9):
    for p in [0.2]:
        data = fillna(iris_x, p)
        pca = PPCA(rng=gu.gen_rng(2))
        pca.fit(data=data, d=2)
        ax = scatter_classes(pca.transform(), iris_y)
        ax.get_figure().savefig('figures/%s-%1.2f.png' % (timestamp, p))
        plt.close('all')
