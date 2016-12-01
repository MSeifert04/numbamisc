Welcome to numbamisc's documentation!
=====================================

Miscellaneous python_ utilities for numpy_ arrays based on numba_ and jinja2_.

.. warning::
   This package is experimental and might undergo backwards-incompatible
   changes in future releases without deprecation period.

.. note::
   The first call to any numba_ function is extremly slow because the function
   needs to be compiled. Subsequent calls will be much faster! If you want to
   compile all functions simply run the test suite. This may take a while.


.. _python: https://www.python.org/
.. _numpy: http://www.numpy.org/
.. _numba: http://numba.pydata.org/
.. _jinja2: http://jinja.pocoo.org/

Contents:

.. toctree::
   :maxdepth: 2

   installation
   filter
   license
   CHANGES
   AUTHORS
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
