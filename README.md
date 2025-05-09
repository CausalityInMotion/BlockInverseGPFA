# BlockInvGPFA

[![Automated Tests](https://github.com/CausalityInMotion/BlockInverseGPFA/actions/workflows/tests.yml/badge.svg)](https://github.com/CausalityInMotion/BlockInverseGPFA/actions/workflows/tests.yml)

This package is an implementation of Gaussian Process Factor Analysis (GPFA) by Byron Yu et al.,
(2009) in python. The code is based on [Elephant](https://elephant.readthedocs.io/en/latest/reference/gpfa.html)'s
python implementation plus additional modules and functional implementations.

## Usage

### Installation
-----------------

- Clone the project locally using 

```bash
$ git clone https://github.com/CausalityInMotion/GPFA_for_sklearn
```

Ensure you are in the working directory of the project.

Then install the project's required packages uisng

```bash
$ pip install -r requirements.txt
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
 
You can install the required packages either using the `docs/requirements.txt` file,
```bash
$ pip install -r docs/requirements.txt
```
or manually by calling
```bash
$ pip install sphinx sphinx-rtd-theme numpydoc nbsphinx
```

Finally, to view the documentation, run

```bash
$ cd docs
$ make html
$ open _build/html/index.html
```

-----------
### Tests
-----------

To run the unittests in the [test](test) folder, use the following command in your command line/terminal:

```bash
$ python -m unittest test.gpfa
$ python -m unittest test.preprocessing
```

## License
Modified BSD License based on Elephant, see [LICENSE.txt](LICENSE.txt) for details.


## Copyright

:copyright: 2021 Brooks M. Musangu and Jan Drugowitsch

## Acknowledgments

See [acknowledgments](docs/acknowledgments.rst).