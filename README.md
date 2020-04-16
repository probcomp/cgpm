# cgpm

[![Build Status](https://travis-ci.org/probcomp/cgpm.svg?branch=master)](https://travis-ci.org/probcomp/cgpm)

The aim of this project is to provide a unified probabilistic programming
framework to express different models and techniques from statistics, machine
learning and non-parametric Bayes. It serves as the primary modeling and
inference runtime system for [bayeslite](https://github.com/probcomp/bayeslite),
an open-source implementation of BayesDB.

Composable generative population models (CGPM) are a computational abstraction
for probabilistic objects. They provide an interface that explicitly
differentiates between the _sampler_ of a random variable from its conditional
distribution and the _assessor_ of its conditional density. By encapsulating
models as probabilistic programs that implement CGPMs, complex models can be
built as compositions of sub-CGPMs, and queried in a model-independent way
using the Bayesian Query Language.

## Installing

### Conda

The easiest way to install cgpm is to use the
[package](https://anaconda.org/probcomp/cgpm) on Anaconda Cloud.
Please follow [these instructions](https://github.com/probcomp/iventure/blob/master/docs/conda.md).

### Manual Build

`cgpm` targets Ubuntu 14.04 and 16.04. The package can be installed by cloning
this repository and following these instructions. It is _highly recommended_ to
install `cgpm` inside of a virtualenv which was created using the
`--system-site-packages` flag.

1. Install dependencies from `apt`, [listed here](https://github.com/probcomp/cgpm/blob/71fe62790f466e9dd2149d0f527c584cce19e70f/docker/ubuntu1604#L4-L14).

2. Retrieve and build the source.

    ```
    % git clone git@github.com:probcomp/cgpm
    % cd cgpm
    % pip install --no-deps .
    ```

3. Verify the installation.

    ```
    % python -c 'import cgpm'
    % cd cgpm && ./check.sh
    ```

## Publications

CGPMs, and their integration as a runtime system for
[BayesDB](probcomp.csail.mit.edu/bayesdb/), are described in the following
technical report:

- __Probabilistic Data Analysis with Probabilistic Programming__.
Saad, F., and Mansinghka, V. [_arXiv preprint, arXiv:1608.05347_](https://arxiv.org/abs/1608.05347), 2017.

Applications of using cgpm and bayeslite for data analysis tasks can be further
found in:

- __Probabilistic Search for Structured Data via Probabilistic Programming and Nonparametric Bayes__.
Saad, F. Casarsa, L., and Mansinghka, V. [_arXiv preprint, arXiv:1704.01087_](https://arxiv.org/abs/1704.01087), 2017.

- __Detecting Dependencies in Sparse, Multivariate Databases Using Probabilistic Programming and Non-parametric Bayes__.
Saad, F., and Mansinghka, V. [_Artificial Intelligence and Statistics (AISTATS)_](http://proceedings.mlr.press/v54/saad17a.html), 2017.

- __A Probabilistic Programming Approach to Probabilistic Data Analysis__.
Saad, F., and Mansinghka, V. [_Advances in Neural Information Processing Systems (NIPS)_](https://papers.nips.cc/paper/6060-a-probabilistic-programming-approach-to-probabilistic-data-analysis.html), 2016.


## Tests

Running `./check.sh` will run a subset of the tests that are considered complete
and stable. To launch the full test suite, including continuous integration
tests, run `py.test` in the root directory. There are more tests in the `tests/`
directory, but those that do not start with `test_` or do start with `disabled_`
are not considered ready. The tip of every branch merged into master __must__
pass `./check.sh`, and be consistent with the code conventions outlined in
[HACKING](HACKING).

To run the full test suite, use `./check.sh --integration tests/`. Note that the
full integration test suite requires installing the C++
[crosscat](https://github.com/probcomp/crosscat) backend.

## License

Copyright (c) 2015-2016 MIT Probabilistic Computing Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
