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
        self.means = None
        self.stds = None

    def _standardize(self, X):
        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")
        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):
        data = np.copy(data)
        Y = np.copy(data.T)

        data[np.isinf(data)] = np.max(data[np.isfinite(data)])
        Y[np.isinf(Y)] = np.max(Y[np.isfinite(Y)])

        valid_series = np.sum(~np.isnan(data), axis=0) >= min_obs
        data = data[:, valid_series].copy()

        valid_series = np.sum(~np.isnan(Y), axis=1) >= min_obs
        Y = Y[valid_series].copy()

        N = data.shape[0]
        D = data.shape[1]

        D2 = Y.shape[0]
        N2 = Y.shape[1]

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)

        mean2 = np.reshape(np.nanmean(Y, axis=1), (-1,1))
        std2 = np.reshape(np.nanstd(Y, axis=1), (-1,1))

        data = self._standardize(data)
        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        data[~observed] = 0

        Y = (Y-mean2)/std2
        observed2 = ~np.isnan(Y)
        missing2 = np.sum(~observed2)
        Y[~observed2] = 0

        assert np.allclose(Y, data.T)

        # Number of components.
        if d is None:
            d = data.shape[1]

        if d is None:
            d = D2

        # Weight matrix.
        if self.C is None:
            C = self.rng.randn(D, d)
        else:
            C = self.C

        CC = np.dot(C.T, C)
        X = np.dot(np.dot(data, C), np.linalg.inv(CC))
        recon = np.dot(X, C.T)
        recon[~observed] = 0
        ss = np.sum((recon - data)**2)/(N*D - missing)

        # Weight matrix.
        W = C
        WW = np.dot(W.T, W)
        Z = np.dot(np.linalg.inv(WW), np.dot(C.T, Y))
        recon2 = np.dot(W, Z)
        recon2[~observed2] = 0
        ss2 = np.sum((recon2 - Y)**2)/(N2*D2 - missing)

        assert np.allclose(W, C)
        assert np.allclose(WW, CC)
        assert np.allclose(X.T, Z)
        assert np.allclose(recon2, recon.T)
        assert np.allclose(ss, ss2)

        v0 = np.inf
        counter = 0

        while True:
            Sx = np.linalg.inv(np.eye(d) + CC/ss)

            Sx2 = np.linalg.inv(np.eye(d) + WW/ss)

            assert np.allclose(Sx, Sx2)

            # E-step.
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, C.T)
                data[~observed] = proj[~observed]
            X = np.dot(np.dot(data, C), Sx) / ss

            ss02 = ss2
            if missing2 > 0:
                proj2 = np.dot(W, Z)
                Y[~observed2] = proj2[~observed2]
            Z = np.dot(Sx, np.dot(W.T, Y)) / ss

            assert np.allclose(ss0, ss02)
            assert np.allclose(proj2, proj.T)
            assert np.allclose(Y, data.T)
            assert np.allclose(X.T, Z)

            import ipdb; ipdb.set_trace()

            # M-step.
            XX = np.dot(X.T, X)
            C = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N*Sx))
            CC = np.dot(C.T, C)
            recon = np.dot(X, C.T)
            recon[~observed] = 0
            ss = (np.sum((recon-data)**2) + N*np.sum(CC*Sx) + missing*ss0)/(N*D)

            # Calculate difference in log likelihood for convergence.
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N*(D*np.log(ss) + np.trace(Sx) - det) \
                + np.trace(XX) - missing*np.log(ss0)
            diff = abs(v1/v0 - 1)
            if verbose:
                print diff
            if (diff < tol) and (counter > 5):
                break

            # Increment counter and proceed.
            counter += 1
            v0 = v1

        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]

        C = np.dot(C, vecs)

        # Attach objects to class.
        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):
        if self.C is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.C)
        return np.dot(data, self.C)

    def _calc_var(self):
        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):
        np.save(fpath, self.C)

    def load(self, fpath):
        assert os.path.isfile(fpath)
        self.C = np.load(fpath)
