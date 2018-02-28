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

import argparse

import numpy as np

from cgpm.utils.config import timestamp

from mnist import prepare_MNIST_data
from vae import VariationalAutoEncoder
from vae_plots import Plot_Manifold_Learning_Result
from vae_plots import Plot_Reproduce_Performance
from vae_runner_check import check_args


IMAGE_SIZE_MNIST = 28
RESULTS_DIR_DEFAULT = './resources/%s.results.d/' % (timestamp())
CKPT_DIR_DEFAULT = './resources/%s.model.ckpt' % (timestamp())

def parse_args():
    """Parsing and configuration."""
    desc = 'Tensorflow implementation of Variational AutoEncoder (VAE)'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_path', type=str, default=RESULTS_DIR_DEFAULT,
        help='File path of output images')
    parser.add_argument('--dim_z', type=int, default='20',
        help='Dimension of latent vector', required=True)
    parser.add_argument('--n_hidden', type=int, default=500,
        help='Number of hidden units in MLP')
    parser.add_argument('--learn_rate', type=float, default=1e-3,
        help='Learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=20,
        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128,
        help='Batch size')
    parser.add_argument('--PRR', type=bool, default=True,
        help='Boolean for plot-reproduce-result')
    parser.add_argument('--PRR_n_img_x', type=int, default=10,
        help='Number of images along x-axis')
    parser.add_argument('--PRR_n_img_y', type=int, default=10,
        help='Number of images along y-axis')
    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
        help='Resize factor for each displayed image')
    parser.add_argument('--PMLR', type=bool, default=False,
        help='Boolean for plot-manifold-learning-result')
    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
        help='Number of images along x-axis')
    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
        help='Number of images along y-axis')
    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
        help='Resize factor for each displayed image')
    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
        help='Range for unifomly distributed latent vector')
    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
        help='Number of samples in order to get distribution of labeled data')
    parser.add_argument('--seed', type=int, default=212,
        help='Random seed')
    parser.add_argument('--num_train', type=int, default=60000,
        help='Number of images in the training set')
    parser.add_argument('--num_test', type=int, default=10000,
        help='Number of images in the test set')
    return check_args(parser.parse_args())


def main(args):
    """Main runner."""

    # Parameters
    # ----------

    RESULTS_DIR = args.results_path
    RNG = np.random.RandomState(args.seed)
    num_train = args.num_train
    num_test = args.num_test

    # Network architecture
    n_hidden = args.n_hidden
    dim_x = IMAGE_SIZE_MNIST**2
    dim_z = args.dim_z

    # Train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR                                  # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x                  # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y                  # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor      # resize factor for each image in a canvas

    PMLR = args.PMLR                                # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x                # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y                # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor    # resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range                # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples            # number of labeled samples to plot a map from input data space to the latent space

    # Load MNIST data
    # ------------------

    (train_data, _train_labels, test_data, test_labels) = \
        prepare_MNIST_data(num_train, num_test)

    # Build the VAE and incorporate the data
    # --------------------------------------

    vae = VariationalAutoEncoder(
        outputs=[0] + range(1, dim_z+1),
        inputs=None,
        dim_x=dim_x,
        dim_z=dim_z,
        n_hidden=n_hidden,
        save_dir=CKPT_DIR_DEFAULT,
        rng=RNG)

    vae.incorporate(train_data)

    # Training
    # --------

    # Plot for reconstruction performance.
    if PRR:
        PRR = Plot_Reproduce_Performance(
            RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST,
            IMAGE_SIZE_MNIST, PRR_resize_factor)
        x_PRR = test_data[0:PRR.n_tot_imgs,:]
        x_PRR_img = np.reshape(x_PRR,
            (PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST))
        PRR.save_images(x_PRR_img, 'input.jpg')

    # Plot for manifold learning result.
    if PMLR and dim_z == 2:
        PMLR = Plot_Manifold_Learning_Result(
            RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST,
            IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)
        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

    min_total_loss = 1e99
    for epoch in xrange(n_epochs):
        total_loss, loss_likelihood, loss_divergence = \
            vae.train_autoencoder(batch_size, learn_rate)

        # Print report every epoc.
        print 'Epoch %d: L_tot %03.2f L_likelihood %03.2f '\
            'L_divergence %03.2f' \
            % (epoch, total_loss, loss_likelihood, loss_divergence)

        # If minimum loss is updated or final epoch, plot results.
        if min_total_loss > total_loss or epoch + 1 == n_epochs:
            min_total_loss = total_loss
            # Plot for reproduce performance.
            if PRR:
                z_recon_PRR = [
                    vae.simulate(None, vae.outputs[1:], {vae.outputs[0]: x})
                    for x in x_PRR
                ]
                x_recon_PRR_dict = [
                    vae.simulate(None, [vae.outputs[0]], z)
                    for z in z_recon_PRR
                ]
                x_recon_PRR = np.asarray([
                    x[vae.outputs[0]] for x in x_recon_PRR_dict
                ])
                # vae.run_x_reconstruct(x_PRR)
                x_recon_PRR_img = x_recon_PRR.reshape(
                    PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                PRR.save_images(x_recon_PRR_img,
                    'PRR_epoch_%02d.jpg' % (epoch,))

            # Plot for manifold learning result.
            if PMLR and dim_z == 2:
                z_grid = (dict(zip(vae.outputs[1:], z)) for z in PMLR.z)

                # x_recon_PMLR = vae.run_x_decode(PMLR.z)
                x_recon_PMLR_dict = [
                    vae.simulate(None, [vae.outputs[0]], z)
                    for z in z_grid
                ]
                x_recon_PMLR = np.asarray([
                    x[vae.outputs[0]] for x in x_recon_PMLR_dict
                ])
                x_recon_PMLR_img = x_recon_PMLR.reshape(
                    PMLR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                PMLR.save_images(x_recon_PMLR_img,
                    'PMLR_epoch_%02d.jpg' % (epoch,))

                # Scatter plot distribution of labeled images
                # z_PMLR = vae.run_z_encode(x_PMLR)
                z_PMLR_dict = [
                    vae.simulate(None, vae.outputs[1:], {vae.outputs[0]: x})
                    for x in x_PMLR
                ]
                z_PMLR = np.asarray([
                    [z[i] for i in vae.outputs[1:]]
                    for z in z_PMLR_dict
                ])
                PMLR.save_scattered_image(z_PMLR, id_PMLR,
                    'PMLR_map_epoch_%02d.jpg' % (epoch,))

        # Confirm can load the graph from disk.
        import json
        import tensorflow as tf
        metadata = vae.to_metadata()
        binary = json.dumps(metadata)
        metadata2 = json.loads(binary)
        tf.reset_default_graph()
        vae = VariationalAutoEncoder.from_metadata(metadata2)
        vae.incorporate(train_data)


if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()

    main(args)
