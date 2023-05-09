.. _installation:

.. role:: bash(code)

   :language: bash

============
Installation
============

Clone the project locally using ``git`` within a `Python virtual environment
<https://docs.python.org/3/library/venv.html>`_:


.. code-block:: bash

    $ git clone https://github.com/CausalityInMotion/GPFA_for_sklearn


Ensure you are in the working directory of the project.

.. code-block:: bash

    $ cd GPFA_for_sklearn

Then install the project's required packages

.. code-block:: bash

    $ pip install -r requirements.txt


You are now set to use the package.

Building the documentation
--------------------------

Building the documentation requires the following packages:

 - `Sphinx <http://www.sphinx-doc.org>`_
 - `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_

Install the required packages using pip:

.. code-block:: bash

   $ pip install sphinx
   $ pip install sphinx-rtd-theme

Finally, to view the documentation run the following commands:

.. code-block:: bash

    $ cd docs
    $ make html
    $ open _build/html/index.html

- The documentation is now.