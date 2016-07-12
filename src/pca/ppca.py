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
        self.Y = None
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

        D = Y.shape[0]
        N = Y.shape[1]

        mean = np.reshape(np.nanmean(Y, axis=1), (-1,1))
        std = np.reshape(np.nanstd(Y, axis=1), (-1,1))

        Y = (Y-mean)/std
        observed = ~np.isnan(Y)
        missing = np.sum(~observed)
        Y[~observed] = 0

        # Number of components.
        if d is None:
            d = D

        # Weight matrix.
        if self.W is None:
            W = self.rng.randn(D, d)
        else:
            W = self.W

        # Weight matrix.
        WW = np.dot(W.T, W)
        X = np.dot(np.linalg.inv(WW), np.dot(W.T, Y))
        recon = np.dot(W, X)
        recon[~observed] = 0
        ss = np.sum((recon - Y)**2)/(N*D - missing)

        v0 = np.inf

        counter = 0

        while True:
            Sx = np.linalg.inv(np.eye(d) + WW/ss)

            # E-step.
            ss0 = ss
            if missing > 0:
                proj = np.dot(W, X)
                Y[~observed] = proj[~observed]
            X = np.dot(Sx, np.dot(W.T, Y)) / ss

            # M-step.
            XX = np.dot(X, X.T)
            W = np.dot(np.dot(Y, X.T), np.linalg.pinv(XX + N*Sx))
            WW = np.dot(W.T, W)
            recon = np.dot(W, X)
            recon[~observed] = 0
            ss = (np.sum((recon-Y)**2) + N*np.sum(WW*Sx) + missing*ss0)/(N*D)

            # Calculate difference in log likelihood for convergence.
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N*(D*np.log(ss) + np.trace(Sx) - det) \
                + np.trace(XX) - missing*np.log(ss0)
            diff2 = abs(v1/v0 - 1)

            print diff2
            if verbose:
                print diff2
            if (diff2 < tol) and (counter > 5):
                break

            v0 = v1

            # Increment counter and proceed.
            counter += 1

        W = orth(W)
        vals, vecs = np.linalg.eig(np.cov(np.dot(W.T, Y)))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:,order]
        vals = vals[order]

        W = np.dot(W, vecs)

        # Attach objects to class.
        self.W = W
        self.Y = Y
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):
        if self.W is None:
            raise RuntimeError('Fit the data model first.')
        data = self.Y if data is None else data.T
        return np.dot(self.W.T, data).T

    def _calc_var(self):
        if self.Y is None:
            raise RuntimeError('Fit the data model first.')
        var = np.nanvar(self.Y, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):
        np.save(fpath, self.W)

    def load(self, fpath):
        assert os.path.isfile(fpath)
        self.W = np.load(fpath)
