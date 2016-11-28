# Built-ins
from __future__ import absolute_import, division, print_function
import doctest

# 3rd party

# This module
import numbamisc


def test_doctests():
    doctest.run_docstring_examples(numbamisc.median_filter, {})
