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

import os

import numpy as np

from scipy.linalg import orth


class PPCA(object):
    def __init__(self, rng):
        self.rng = rng
        self.data = None
        self.C = None
        self.W = None
        self.means = None
        self.stds = None

    def _standardize(self, X):
        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")
        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):
        Y = np.copy(data.T)

        Y[np.isinf(Y)] = np.max(Y[np.isfinite(Y)])

        valid_series = np.sum(~np.isnan(Y), axis=1) >= min_obs
        Y = Y[valid_series].copy()

        D2 = Y.shape[0]
        N2 = Y.shape[1]

        mean2 = np.reshape(np.nanmean(Y, axis=1), (-1,1))
        std2 = np.reshape(np.nanstd(Y, axis=1), (-1,1))

        Y = (Y-mean2)/std2
        observed2 = ~np.isnan(Y)
        missing2 = np.sum(~observed2)
        Y[~observed2] = 0

        # Number of components.
        if d is None:
            d = D2

        # Weight matrix.
        if self.W is None:
            W = self.rng.randn(D2, d)
        else:
            W = self.W

        # Weight matrix.
        WW = np.dot(W.T, W)
        Z = np.dot(np.linalg.inv(WW), np.dot(W.T, Y))
        recon2 = np.dot(W, Z)
        recon2[~observed2] = 0
        ss2 = np.sum((recon2 - Y)**2)/(N2*D2 - missing2)

        v02 = np.inf

        counter = 0

        while True:
            Sx2 = np.linalg.inv(np.eye(d) + WW/ss2)

            # E-step.
            ss02 = ss2
            if missing2 > 0:
                proj2 = np.dot(W, Z)
                Y[~observed2] = proj2[~observed2]
            Z = np.dot(Sx2, np.dot(W.T, Y)) / ss2

            # M-step.
            ZZ = np.dot(Z, Z.T)
            W = np.dot(np.dot(Y, Z.T), np.linalg.pinv(ZZ + N2*Sx2))
            WW = np.dot(W.T, W)
            recon2 = np.dot(W, Z)
            recon2[~observed2] = 0
            ss2 = (np.sum((recon2-Y)**2) + N2*np.sum(WW*Sx2) + missing2*ss02)/(N2*D2)

            # Calculate difference in log likelihood for convergence.
            det2 = np.log(np.linalg.det(Sx2))
            if np.isinf(det2):
                det2 = abs(np.linalg.slogdet(Sx2)[1])
            v12 = N2*(D2*np.log(ss2) + np.trace(Sx2) - det2) \
                + np.trace(ZZ) - missing2*np.log(ss02)
            diff2 = abs(v12/v02 - 1)

            print diff2
            if verbose:
                print diff2
            if (diff2 < tol) and (counter > 5):
                break

            v02 = v12

            # Increment counter and proceed.
            counter += 1

        W = orth(W)
        vals2, vecs2 = np.linalg.eig(np.cov(np.dot(W.T, Y)))
        order2 = np.flipud(np.argsort(vals2))
        vecs2 = vecs2[:,order2]
        vals2 = vals2[order2]

        W = np.dot(W, vecs2)


        # Attach objects to class.
        self.W = W
        self.Y = Y
        self.eig_vals2 = vals2
        self._calc_var()

    def transform(self, data=None):
        if self.W is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.W.T, self.Y).T
        return np.dot(self.W, data).T

    def _calc_var(self):
        if self.Y is None:
            raise RuntimeError('Fit the data model first.')

        # data = self.data.T

        # variance calc
        var = np.nanvar(self.Y, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals2.cumsum() / total_var

    def save(self, fpath):
        np.save(fpath, self.W)

    def load(self, fpath):
        assert os.path.isfile(fpath)
        self.W = np.load(fpath)
