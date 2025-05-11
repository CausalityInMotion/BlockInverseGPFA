.. _installation:

====================
Installation Guide
====================

Clone the Project Locally
-------------------------

Begin by cloning the project repository locally. It is strongly recommended to do this inside a `Python virtual environment`_:

.. code-block:: bash

    $ git clone https://github.com/CausalityInMotion/BlockInverseGPFA.git
    $ cd BlockInverseGPFA

Install the Package
-------------------

Use `pip` to install the package and its core dependencies:

.. code-block:: bash

    $ pip install .

If you also want to run tests or build documentation, you can install the appropriate optional dependencies:

- For testing:

  .. code-block:: bash

      $ pip install .[test]

- For documentation:

  .. code-block:: bash

      $ pip install .[docs]

Building the Documentation
--------------------------

To build the documentation, you will need the following packages:

- Sphinx_: A documentation generator.
- `Read the Docs Sphinx Theme`_: A theme for Sphinx that provides an elegant and readable documentation format.

After installing the documentation dependencies, generate the HTML documentation as follows:

.. code-block:: bash

    $ cd docs
    $ make html

To view the generated documentation in your browser:

.. code-block:: bash

    $ open _build/html/index.html  # On macOS

    # OR
    $ xdg-open _build/html/index.html  # On Linux

    # OR
    $ start _build/html/index.html  # On Windows (in cmd)

Your documentation is now available for reference.

Alternatively, you can view the latest published documentation online:

📖 https://blockinversegpfa.readthedocs.io/en/latest/

About This Guide
----------------

This guide provides step-by-step instructions for installing the BlockInvGPFA package, running its test suite, and building its documentation using Python packaging standards.

.. _Python virtual environment: https://docs.python.org/3/library/venv.html
.. _Sphinx: http://www.sphinx-doc.org
.. _Read the Docs Sphinx Theme: https://sphinx-rtd-theme.readthedocs.io/en/stable/
