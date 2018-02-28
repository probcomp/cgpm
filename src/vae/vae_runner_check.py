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

import glob
import os


def check_args(args):
    """Checking arguments."""

    # --results_path
    try:
        os.mkdir(args.results_path)
    except OSError:
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --dim-z
    try:
        assert args.dim_z > 0
    except Exception:
        print 'dim_z must be positive integer'
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except Exception:
        print 'number of hidden units must be larger than one'

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except Exception:
        print 'learning rate must be positive'

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except Exception:
        print 'number of epochs must be larger than or equal to one'

    # --batch_size
    try:
        assert args.batch_size >= 1
    except Exception:
        print 'batch size must be larger than or equal to one'

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except Exception:
        print 'PRR must be boolean type'
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except Exception:
            print 'PRR : number of images along each axis must be larger'\
                ' than or equal to one'

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except Exception:
            print 'PRR : resize factor for each displayed image '\
                ' must be positive'

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except Exception:
        print 'PMLR must be boolean type'
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except Exception:
            print 'PMLR : dim_z must be two'
        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except Exception:
            print 'PMLR : number of images along each axis must be larger '\
                'than or equal to one'
        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except Exception:
            print 'PMLR : resize factor for each displayed image must'\
                ' be positive'
        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except Exception:
            print 'PMLR : range for uniformly distributed latent vector'\
                ' must be positive'
        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except Exception:
            print 'PMLR : Number of samples in order to get distribution of'\
                ' labeled data must be large enough'

    # --seed
    try:
        assert args.seed >= 0
    except Exception:
        print 'seed must be positive'

    # --num-train
    try:
        assert args.num_train > 0
    except Exception:
        print 'num_train must be positive'

    # --num-test
    try:
        assert args.num_test > 0
    except Exception:
        print 'num_test must be positive'

    return args
