# gpmcc

Implementation of [crosscat](http://probcomp.csail.mit.edu/crosscat/) from
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

A plot similar to ![../../images/one_view.png](../../images/one_view.png) should
appear.
