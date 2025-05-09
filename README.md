# BlockInvGPFA

[![Automated Tests](https://github.com/CausalityInMotion/BlockInverseGPFA/actions/workflows/tests.yml/badge.svg)](https://github.com/CausalityInMotion/BlockInverseGPFA/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/blockinversegpfa/badge/?version=latest)](https://blockinversegpfa.readthedocs.io/en/latest/)

This package provides a Python implementation of **Gaussian Process Factor Analysis (GPFA)** that incorporates a novel approach to efficiently handle variable-length time series data. Building on the original method by [Byron Yu et al. (2009)](https://papers.nips.cc/paper/2009/hash/6c1b887a379c4f0c2c621f305d15f6b0-Abstract.html), this implementation introduces a **block-matrix–based inversion strategy** that reuses kernel computations across trials of different lengths.

The package also includes **scikit-learn–compatible API** for integration into existing ML workflows such as:
- A **Modular Preprocessing** — Separates data preprocessing from model logic via a dedicated `EventTimesToCounts` transformer.
  - Accepts standard array-like inputs (not `Neo` objects), simplifying integration with other tools.
  - Follows scikit-learn's transformer–estimator interface for clean, reusable workflows.
- A new **variance-explained metric** to evaluate GPFA model fits.

This implementation is adapted from [Elephant](https://elephant.readthedocs.io/en/latest/reference/gpfa.html)’s GPFA codebase with substantial modifications to improve performance, modularity, and usability in Python-based pipelines.


## Usage

### Installation
-----------------

Install directly from GitHub using `pip`:

```bash
$ pip install git+https://github.com/CausalityInMotion/BlockInverseGPFA.git
```
Or clone the repo and install locally:

```bash
$ git clone https://github.com/CausalityInMotion/BlockInverseGPFA.git
$ cd BlockInverseGPFA
$ pip install . # or pip install .[test,docs] to include optional dependencies
```

You are now set to use the package.

------------------------------
### Building the documentation
------------------------------

Building the documentation requires the following packages:

 - [Sphinx](http://www.sphinx-doc.org)
 - [Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)
 - [numpydoc](https://numpydoc.readthedocs.io/)
 - [Jupyter Notebook Tools for Sphinx](https://nbsphinx.readthedocs.io/)
 
You can install the required documentation dependencies by:
```bash
$ pip install .[docs]
```
or manually by calling
```bash
$ pip install sphinx sphinx-rtd-theme numpydoc nbsphinx
```

Finally, to view the documentation locally, run

```bash
$ cd docs
$ make html
$ open _build/html/index.html
```
or view them online:
[BlockInvGPFA docs](https://blockinversegpfa.readthedocs.io/en/latest/)

A detailed walkthrough of the package — including how to fit the model to real neural data — is available in the Jupyter notebook example: (link)

-----------
### Tests
-----------

To run the full test suite in the [test](test) folder, use:

```bash
$ pip install .[test]
$ pytest test/
```
Tests are automatically run via [GitHub Actions](https://github.com/CausalityInMotion/BlockInverseGPFA/actions/new) on every push and pull request.

## License
Modified BSD License based on Elephant, see [LICENSE.txt](LICENSE.txt) for details.


## Copyright

:copyright: 2021-2025 Brooks M. Musangu and Jan Drugowitsch

## Acknowledgments

See [acknowledgments](docs/acknowledgments.rst).