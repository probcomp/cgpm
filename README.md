# cgpm

The aim of this project is to provide a unified probabilistic programming
framework to express different models and techniques from statistics, machine
learning and non-parametric Bayes. It serves as the primary modeling and
inference runtime system for [BayesDB](https://github.com/probcomp/bayeslite).

Composable generative population models (CGPM) are a computational abstraction
for probabilistic objects. They provide an interface that explicitly
differentiates between the _sampler_ of a random variable from its conditional
distribution and the _assessor_ of its conditional density. By encapsulating
models as probabilistic programs that implement CGPMs, complex models can be
built as compositions of sub-CGPMs, and queried in a model-independent way
using the Bayesian Query Language.

## Reference

CGPMs, and their integration as a runtime system for
[BayesDB](probcomp.csail.mit.edu/bayesdb/), are described in the following
technical report:

> Probabilistic Data Analysis with Probabilistic Programming. Saad, F. and Mansinghka, V.
> [arXiv:1608.05347](https://arxiv.org/abs/1608.05347).

A shorter version of the technical report can be found in:

> A Probabilistic Programming Approach To Probabilistic Data Analysis.
> Saad, F. and Mansinghka, V. Neural Information Processing Systems, 2016.
> [PDF](https://papers.nips.cc/paper/6060-a-probabilistic-programming-approach-to-probabilistic-data-analysis).

## Installing

For developers: instructions for installing `cgpm` and related probcomp software
can be found in the [iventure project](https://github.com/probcomp/iventure/blob/master/docs/virtualenvs.md).

## Tests

Running `./check.sh` will run a subset of the tests that are considered complete
and stable. To launch the full test suite, including continuous integration
tests, run `py.test` in the root directory. There are more tests in the `tests/`
directory, but those that do not start with `test_` or do start with `disabled_`
are not considered ready. The tip of every branch merged into master __must__
pass `./check.sh`, and be consistent with the code conventions outlined in
[HACKING](HACKING).

To run the full test suite, use `./check.sh --integration tests/`.

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
