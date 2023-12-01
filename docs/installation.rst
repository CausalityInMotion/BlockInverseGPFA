.. _installation:

====================
Installation Guide
====================

Clone the Project Locally
--------------------------

Begin by cloning the project's repository locally using the `git` command within a Python virtual environment. Ensure that you have `Python virtual environment`_ installed and activated before proceeding. Open a terminal and run the following commands:

.. code-block:: bash

    $ git clone https://github.com/CausalityInMotion/GPFA_for_sklearn
    $ cd GPFA_for_sklearn

Install Required Packages
-------------------------

Once you are in the project's working directory, install the required Python packages using `pip` and the `requirements.txt` file:

.. code-block:: bash

    $ pip install -r requirements.txt

Your environment is now properly configured to use the GPFA package.

Building the Documentation
--------------------------

To build the documentation, you will need the following Python packages:

- `Sphinx`_: A documentation generator.
- `Read the Docs Sphinx Theme`_: A theme for Sphinx that provides an elegant and readable documentation format.

You can install these packages using `pip`:

.. code-block:: bash

   $ pip install sphinx
   $ pip install sphinx-rtd-theme

After installing the required packages, navigate to the "docs" directory:

.. code-block:: bash

    $ cd docs

To generate the documentation in HTML format, run the following command:

.. code-block:: bash

    $ make html

To view the generated documentation, open the HTML index page in your web browser:

.. code-block:: bash

    $ open _build/html/index.html

Your documentation is now available for reference.

This guide provides step-by-step instructions for installing the GPFA package and generating its documentation.

.. _Python virtual environment: https://docs.python.org/3/library/venv.html
.. _Sphinx: http://www.sphinx-doc.org
.. _Read the Docs Sphinx Theme: https://sphinx-rtd-theme.readthedocs.io/en/stable/
