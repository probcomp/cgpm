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

import matplotlib;
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import imresize
from scipy.misc import imsave


class Plot_Reproduce_Performance(object):

    def __init__(self, dirname, n_img_x=8, n_img_y=8, img_w=28, img_h=28,
            resize_factor=1.0):
        self.dirname = dirname
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h
        assert resize_factor > 0
        self.resize_factor = resize_factor

    def save_images(self, images, fname):
        dim = (self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        images = np.reshape(images, dim)
        merged = merge_images(images, [self.n_img_y, self.n_img_x],
            self.resize_factor)
        figname = os.path.join(self.dirname, fname)
        imsave(figname, merged)


class Plot_Manifold_Learning_Result(object):
    def __init__(self, dirname, n_img_x=20, n_img_y=20, img_w=28, img_h=28,
            resize_factor=1.0, z_range=4):
        self.dirname = dirname
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h
        assert resize_factor > 0
        self.resize_factor = resize_factor
        assert z_range > 0
        self.z_range = z_range
        z = np.rollaxis(
            np.mgrid[
                self.z_range:-self.z_range:self.n_img_y * 1j,
                self.z_range:-self.z_range:self.n_img_x * 1j],
            0, 3)
        self.z = z.reshape([-1, 2])

    def save_images(self, images, fname):
        dim = self.n_img_x*self.n_img_y, self.img_h, self.img_w
        images = np.reshape(images, dim)
        merged = merge_images(images, [self.n_img_y, self.n_img_x],
            self.resize_factor)
        figname = os.path.join(self.dirname, fname)
        imsave(figname, merged)

    def save_scattered_image(self, z, ids, fname):
        fig, ax = plt.subplots(figsize=(8, 6))
        N = 10
        points = ax.scatter(z[:,0], z[:,1], c=np.argmax(ids, 1),
            marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        fig.colorbar(points, ticks=range(N))
        ax.set_xlim([-self.z_range-2, self.z_range+2])
        ax.set_ylim([-self.z_range-2, self.z_range+2])
        ax.grid(True)
        figname = os.path.join(self.dirname, fname)
        fig.savefig(figname)


def merge_images(images, size, resize_factor):
    h, w = images.shape[1], images.shape[2]
    h_ = int(h * resize_factor)
    w_ = int(w * resize_factor)
    img = np.zeros((h_*size[0], w_*size[1]))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        image_ = imresize(image, size=(w_, h_), interp='bicubic')
        img[j*h_:j*h_+h_, i*w_:i*w_+w_] = image_
    return img


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
