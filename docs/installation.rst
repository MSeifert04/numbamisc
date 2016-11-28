Installation
------------

The `numbamisc` package is a pure python package supporting the
Python versions:

- Python 2.7
- Python 3.3+


Using pip
^^^^^^^^^

and can be installed with ``pip`` [0]_:

- ``pip install numbamisc``


or to install the development version:

- ``pip install git+https://github.com/MSeifert04/numbamisc.git@master``


Using conda
^^^^^^^^^^^

It can by installed with ``conda`` [2]_ from the ``conda-forge`` channel:

- ``conda install -c conda-forge numbamisc``


Manual installation
^^^^^^^^^^^^^^^^^^^

or download the development version from ``git`` [1]_ and install it:

- ``git clone https://github.com/MSeifert04/numbamisc.git``
- ``cd numbamisc``
- ``python setup.py install``

with the clone from ``git`` one can also run:

- ``python setup.py test`` (run the test suite)
- ``python setup.py build_sphinx`` (local documentation build)

Dependencies
^^^^^^^^^^^^

Installation:

- Python2 2.7 or Python3 3.3+
- setuptools
- numpy
- scipy
- jinja2
- numba


Tests:

- pytest
- pytest-runner


Documentation:

- sphinx
- numpydoc


References
~~~~~~~~~~

.. [0] https://github.com/MSeifert04/numbamisc
.. [1] https://pypi.python.org/pypi/numbamisc
.. [2] https://www.continuum.io/
