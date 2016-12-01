# Built-ins
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import pytest
from scipy import ndimage

# This module
import numbamisc


@pytest.mark.parametrize("kind, cmp_ft",
                         [(numbamisc.median_filter, ndimage.median_filter),
                          (numbamisc.average_filter, None),
                          (numbamisc.sum_filter, None),
                          (numbamisc.median_filter_weigthed, None),
                          (numbamisc.min_filter, ndimage.minimum_filter),
                          (numbamisc.max_filter, ndimage.maximum_filter),
                          ])
@pytest.mark.parametrize("dtype",
                         [np.float_, np.int_])
@pytest.mark.parametrize("ndim",
                         [1, 2, 3, 4, 5])
@pytest.mark.parametrize("border",
                         ['ignore', 'nearest', 'wrap', 'reflect', 'mirror'])
@pytest.mark.parametrize("nan",
                         [False, True])
def test_no_error(kind, dtype, ndim, border, nan, cmp_ft):
    arr = (np.random.random((3, ) * ndim) * 10).astype(dtype)
    mask = np.zeros((3, ) * ndim, bool)
    kernel = np.ones((3, ) * ndim, np.int_)
    res = kind(arr, kernel, mask, mode=border, ignore_nan=nan)
    if border != 'ignore' and not nan:
        if cmp_ft is not None:
            ref = cmp_ft(arr, footprint=kernel, mode=border)
        elif kind is numbamisc.sum_filter:
            ref = ndimage.convolve(arr, kernel, mode=border)
        elif kind is numbamisc.average_filter:
            ref = ndimage.convolve(arr, kernel, mode=border) / np.sum(kernel)
        else:
            return
        np.testing.assert_array_almost_equal(res[0], ref)
