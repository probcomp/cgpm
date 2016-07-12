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
        # Random seed.
        self.rng = rng
        # Principal component vectors.
        self.W = None
        # Dataset and metadata.
        self.Y = None
        self.mean = None
        self.std = None
        # Eigenvalues and eigenvectors of covariance.
        self.eig_vals = None
        self.eig_vecs = None

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):
        # Defensive copy.
        Y = np.copy(data.T)

        # Convert infinite values to their maximum.
        Y[np.isinf(Y)] = np.max(Y[np.isfinite(Y)])

        # Drop dimensions which have <= min_obs entries.
        valid_series = np.sum(~np.isnan(Y), axis=1) >= min_obs
        Y = Y[valid_series].copy()

        # Extract number of dimensions, and number of data points.
        D, N = Y.shape

        # Compute dataset statistics.
        self.mean = np.reshape(np.nanmean(Y, axis=1), (-1,1))
        self.std = np.reshape(np.nanstd(Y, axis=1), (-1,1))

        # Standard the dataset.
        Y = self._standardize(Y)

        # Replace nan with zeros.
        observed = ~np.isnan(Y)
        missing = np.sum(~observed)
        Y[~observed] = 0

        # Number of principal component vectors.
        if d is None:
            d = D

        # Matrix of principal componetn vectors.
        if self.W is None:
            W = self.rng.randn(D, d)
        else:
            W = self.W

        # Initial values of params and latents.
        WW = np.dot(W.T, W)
        X = np.dot(np.linalg.inv(WW), np.dot(W.T, Y))
        recon = np.dot(W, X)
        recon[~observed] = 0
        ss = np.sum((recon - Y)**2)/(N*D - missing)

        # Initial log likelihood.
        ll0 = np.inf

        # Iteration counter.
        counter = 0

        while True:
            # Covariance matrix for Gaussian p(X|Y,W).
            Sx = np.linalg.inv(np.eye(d) + WW/ss)

            # E-step.
            ss0 = ss
            if missing:
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

            # Calculate log likelihood.
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            ll1 = N*(D*np.log(ss) + np.trace(Sx) - det) \
                + np.trace(XX) - missing*np.log(ss0)

            # Break at convergence.
            delta = abs(ll1/ll0 - 1)
            if verbose:
                print delta
            if (delta < tol) and (counter > 5):
                break

            # Increment counter and proceed.
            ll0 = ll1

            # Increment counter.
            counter += 1

        W = orth(W)
        eig_vals, eig_vecs = np.linalg.eig(np.cov(np.dot(W.T, Y)))
        order = np.flipud(np.argsort(eig_vals))
        eig_vecs = eig_vecs[:,order]
        eig_vals = eig_vals[order]

        W = np.dot(W, eig_vecs)

        # Attach objects to class.
        self.W = W
        self.Y = Y
        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs
        self._calc_var()

    def transform(self, data=None):
        if self.W is None:
            raise RuntimeError('Fit the data model first.')
        data = self.Y if data is None else data.T
        return np.dot(self.W.T, data).T

    def _standardize(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("Fit model first")
        return (X - self.mean) / self.std

    def _calc_var(self):
        if self.Y is None:
            raise RuntimeError('Fit model first.')
        var = np.nanvar(self.Y, axis=1)
        total_var = np.sum(var)
        return np.cumsum(self.eig_vals) / total_var

    def save(self, fpath):
        np.save(fpath, self.W)

    def load(self, fpath):
        assert os.path.isfile(fpath)
        self.W = np.load(fpath)
