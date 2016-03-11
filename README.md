# gpmcc

Implementation of [crossat](http://probcomp.csail.mit.edu/crosscat/) from
the lens of generative population models (GPMs). The goal is to express the
hierarchial generative process that defines crosscat as a composition of
modules that follow the GPM interface.

## Research Goals

gpmcc aims to implement all the features that exist in current crosscat
implementations, as well as new features not available in vanilla crosscat.
Key ideas on the development roadmap are:

- Interface that permits key constructs of the Metamodeling Language (MML),
  such as:
  - Suggesting column dependencies.
  - Suggesting row dependencies, with respect to a subset of columns.
  - Sequential incorporate/unincorporate of (partial) observations
    interleaved with analysis.
  - Targeted analysis over the crosscat inference kernels.
  - Column-specific datatype constraints (`REAL`, `POSITIVE`,
  `IN-RANGE(min,max)`, `CATEGORICAL`, `ORDINAL`, etc).

- Sequential Monte Carlo (SMC) implementation of the posterior inference
  algorithm described in [Mansinghka, et
  al.](http://arxiv.org/pdf/1512.01272.pdf) Section 2.4, as opposed to
  observe-all then Gibbs forever.

- Interface for the Bayesian Query Language (BQL) and
  [bayeslite](https://github.com/probcomp/bayeslite) integration, with new
  BQL additions such as:
  - Conditional mutual information.
  - KL-divergence of predictive distribution against synthetic GPMs.
  - Marginal likelihood estimates of real datasets.

- Interface for foreign GPMs that are jointly analyzed with crosscat.
  Current implementations only allow foreign GPMs to be composed at query,
  not analysis, time.

- Subsampling, where each model is responsible for a subset of data from an
  overlapping partition of the overall dataset.

- Multiprocessing for analysis. Distributed?

- Several DistributionGpms for the different MML data-types, not just
  Normal and Multinomial.

## Required Modules
- numpy
- scipy
- matplotlib

## Installing
```
pip install .
```

## Static Example

The simplest example is creating a synthetic dataset where each variable is
a mixture of one of the available DistributionGpms. Try

```
$ python -i examples/one_view.py
```

A plot similar to ![images/one_view.png](images/one_view.png) should appear.

## Interactive Example (Experimental)

Single-particle SMC is currently available for a dataset with a single
variable. To view an interactive example, try the following

```
$ cd experiments/univariate_crp
$ python -i crp.py
```

Click on the graph to produce observations and watch, the Gibbs kernel cycle
through the hypothesis space

![images/smc.gif](images/smc.gif)

The values printed in the console after each click are estimates of the
marginal-log-likelihood of observations, based on the single particle
weight. The following output

```
Observation 8.000000: 0.209677
[-8.0740236375201153]
```

means the eigth observation is 0.209677, and the estimated marginal
log-liklelihood is -8.0740236375201153.

## Tests

Running check.sh will run the tests that are considered complete and
stable. There are more tests in the tests/ directory, but those that
do not start with ```test_``` or do start with ```disabled_``` are not
considered ready.

See the open [issue](https://github.com/probcomp/gpmcc/issues/8).

## Acknowledgements

This repository was originally forked off
[BaxCat](https://github.com/BaxterEaves/BaxCat/). Most of the source has been
significantly rewritten, and copyright rights have been transferred with
permission.

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
