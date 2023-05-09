.. _contribute:

============
Contribution
============

Contributions to the GPFA package are highly welome on all fronts, including enhancement,
bug fixes and documentation.

In order to maintain a particular level of code quality, please read this guide 
throughly.

Code Style
----------

We adhere to the `PEP 8 -- Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_.

Code Documentation
------------------
Documentate your code, please. There shoul be a comment for each module, class and function.
In the current package, we use `Numpydoc Style 
<https://numpydoc.readthedocs.io/en/latest/format.html>`_, and the example can be found `here 
<https://numpydoc.readthedocs.io/en/latest/example.html>`_. Furthermore, please update the
documentation you are reading if there are significant changes. It was created using `Sphinx 
<http://www.sphinx-doc.org>`_, and the source files may be found in the ``./docs`` directory.

Code tests
----------
To help identify bugs or errors during the development process, please run the ``test`` modules
located in ``./test`` directory. Please see the ``README.md`` on how to run the ``test`` modules. 
Furthermore, as far as possible, include tests for any module, class or funciton contribution.

.. add Folking, pushing and creating pull request