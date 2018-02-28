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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
import tensorflow as tf

from scipy import ndimage
from six.moves import urllib


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = 'data'

# Parameters for MNIST.
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def maybe_download(filename):
    """Download the data, unless it is already here."""
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print ('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images, norm_shift=False, norm_scale=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5]."""
    print ('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE* IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        if norm_shift:
            data = data - (PIXEL_DEPTH / 2.0)
        if norm_scale:
            data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data),labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


def expend_training_data(images, labels):
    """Augment training data."""
    expanded_images = []
    expanded_labels = []
    j = 0
    for x, y in zip(images, labels):
        j = j + 1
        if j%100 == 0:
            print ('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))
        # Register original data.
        expanded_images.append(x)
        expanded_labels.append(y)
        # Get a value for the background.
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x) # this is regarded as background's value
        image = numpy.reshape(x, (-1, 28))
        for _i in xrange(4):
            # rotate the image with random degree.
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            # Shift the image with random distance.
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)
            # Register new training data.
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)
    # Images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken.
    expanded_train_total_data = numpy.concatenate(
        (expanded_images, expanded_labels), axis=1)
    # XXX Remove call to numpy.random.
    numpy.random.shuffle(expanded_train_total_data)
    return expanded_train_total_data


def prepare_MNIST_data(num_train, num_test, norm_shift=False, norm_scale=True,
        data_augmentation=False):
    """Return MNIST datasets as numpy arrays."""
    # Fetch the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # Extract training/validation datat into numpy arrays.
    train_data = extract_data(train_data_filename, num_train,
        norm_shift, norm_scale)
    train_labels = extract_labels(train_labels_filename, num_train)
    # train_data = train_data_all[VALIDATION_SIZE:, :]
    # train_labels = train_labels_all[VALIDATION_SIZE:,:]
    # validation_data = train_data_all[:VALIDATION_SIZE, :]
    # validation_labels = train_labels_all[:VALIDATION_SIZE,:]
    # Extract test data.
    test_data = extract_data(test_data_filename, num_test,
        norm_shift, norm_scale)
    test_labels = extract_labels(test_labels_filename, num_test)
    # Concatenate train_data & train_labels for random shuffle
    if data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)
    # Separate the training data and training labels again.
    train_data_return = train_total_data[:,:-NUM_LABELS]
    train_labels_return = train_total_data[:,-NUM_LABELS:]

    return \
        train_data_return, \
        train_labels_return, \
        test_data, \
        test_labels
        # validation_data, \
        # validation_labels, \
