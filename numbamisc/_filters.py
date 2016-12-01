from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from scipy import ndimage as scipy_ndimage

try:
    from ._genfilters import filters
except ImportError:
    from .utils import generate

    current_dir = os.path.dirname(__file__)
    target_dir = current_dir

    try:
        os.makedirs(target_dir)
    except OSError:
        pass

    with open(os.path.join(target_dir, '_genfilters.py'), 'w') as fobj:
        fobj.write(generate(maxndim=5))

    from ._genfilters import filters


__all__ = ['median_filter', 'median_filter_weigthed',
           'min_filter', 'max_filter',
           'average_filter', 'sum_filter']


ParameterNotSpecified = object()


def _convert_to_native_bytorder(array):
    """Convert a numpy array with endian dtype to an array of native dtype.

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array to convert. Must be convertable to a numpy.ndarray.

    Returns
    -------
    converted_array : `numpy.ndarray`
        The converted array. If the array already was in native dtype the array
        is simple returned without making a copy.

    Notes
    -----
    This function is essential when using ``numba`` because it cannot process
    small/big endian arrays. At least it seems like that.
    """
    array = np.asarray(array)
    return array.astype(array.dtype.type, copy=False)


def _getdatamask(data, mask):
    # Try to just extract a mask from the data. If it fails with an
    # AttributeError and no mask was specified create an empty boolean array.
    try:
        mask2 = data.mask
    except AttributeError:
        mask2 = None
    else:
        # In case we sucessfully used the mask of the data we either have a
        # numpy.ma.MaskedArray or similar.
        data = data.data

    # Only in case no explicit mask was given use the one extracted from the
    # data.
    if mask is ParameterNotSpecified:
        mask = mask2

    data = _convert_to_native_bytorder(data)

    # Check if the shape is the same. There might be cases where the
    # array contained a mask attribute but the mask has a different shape
    # than the data!
    if mask is not None:
        mask = _convert_to_native_bytorder(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')

    return data, mask


def _getkernel(kernel, ndim):
    # Create a kernel from an integer (width in all dimensions)
    if isinstance(kernel, int):
        kernel = np.ones((kernel, ) * ndim, int)

    # Create a kernel from a tuple of integer:
    if isinstance(kernel, tuple):
        kernel = np.ones(kernel, int)

    # It is possible that the kernel is an astropy kernel, in that case it has
    # an attribute "array" and we use that one:
    kernel = _convert_to_native_bytorder(getattr(kernel, 'array', kernel))

    # kernel must have the same number of dimensions
    if kernel.ndim != ndim:
        raise ValueError('data and kernel must have the same number of '
                         'dimensions.')

    return kernel


def _process_input(data, kernel, mask):
    """Checks prerequisites and loads/applies the appropriate function.
    """
    data, mask = _getdatamask(data, mask)

    kernel = _getkernel(kernel, data.ndim)

    return data, kernel, mask


def median_filter(data, kernel, mask=ParameterNotSpecified,
                  mode='ignore', ignore_nan=False):
    """Median based convolution of some data by ignoring masked values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to convolve.

    kernel : :class:`~numpy.ndarray`, :class:`~astropy.convolution.Kernel`
        The kernel for the convolution. One difference from normal convolution
        is that the actual values of the kernel do not matter, except when it
        is zero then it won't use the element for the median computation.
        Each axis of the kernel must be odd.

    mask : :class:`~numpy.ndarray`, optional
        Masked values in the ``data``. Elements where the mask is equivalent to
        1 (also ``True``) are interpreted as masked and are ignored during the
        convolution. If not given use the mask of the data or if it has no mask
        either just use `scipy.ndimage.median_filter`.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The median filtered array.

    See also
    --------
    scipy.ndimage.median_filter : Fast n-dimensional convolution \
        without masks.

    Notes
    -----
    If the ``data`` parameter has a ``mask`` attribute then ``data.data``
    is interpreted as ``data`` and ``array.mask`` as ``mask`` parameter.
    This allows using :class:`~numpy.ma.MaskedArray` objects as ``data``
    parameter.

    If an explicit ``mask`` is given (even if it is ``None``) an implicit
    mask is ignored.

    No border handling is possible, if the kernel extends beyond the image
    these *outside* values are treated as if they were masked.

    Examples
    --------
    A simple example using a masked array::

        >>> from numbamisc import median_filter
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> d, m = median_filter(data, [1,1,1])
        >>> d
        array([ 1. ,  1.5,  1.5,  1.5])
        >>> m
        array([False, False, False, False], dtype=bool)

    Support for arbitarly dimensional arrays and masks is also implemented::

        >>> data = np.arange(9).reshape(3, 3)
        >>> data[1, 1] = 100
        >>> mask = np.zeros((3, 3), dtype=bool)
        >>> mask[1, 1] = 1
        >>> d, m = median_filter(data, np.ones((3,3)), mask)
        >>> d
        array([[ 1.,  2.,  2.],
               [ 3.,  4.,  5.],
               [ 6.,  6.,  7.]])
        >>> m
        array([[False, False, False],
               [False, False, False],
               [False, False, False]], dtype=bool)

    And another example::

        >>> data = np.arange(27).reshape(3, 3, 3)
        >>> data[0, 0, 0] = 10000
        >>> mask = np.zeros((3, 3, 3))
        >>> mask[0, 0, 0] = 1
        >>> d, m = median_filter(data, np.ones((3, 3, 3)), mask)
        >>> d
        array([[[  9. ,   9. ,   7.5],
                [  9. ,   9. ,   9. ],
                [  9.5,  10. ,  10.5]],
        <BLANKLINE>
               [[ 12. ,  12. ,  12. ],
                [ 13. ,  13.5,  13.5],
                [ 14. ,  14.5,  15. ]],
        <BLANKLINE>
               [[ 15.5,  16. ,  16.5],
                [ 17. ,  17.5,  18. ],
                [ 18.5,  19. ,  19.5]]])

    Explictly using kernel elements to zero excludes those elements for the
    convolution::

        >>> data = np.ma.array([1, 1000, 2, 1], mask=[0, 1, 0, 0])
        >>> d, m = median_filter(data, [1, 0, 0])
        >>> d
        array([ nan,   1.,  nan,   2.])
        >>> m
        array([ True, False,  True, False], dtype=bool)

    Here only the left element is used for the convolution. For the first
    element the left one is outside the data and for the third element the
    convolution element is masked so both of them result in ``NaN``.
    """
    data, kernel, mask = _process_input(data, kernel, mask)

    # In case we have no mask (None) simply use scipy.median_filter
    if mask is None:
        return scipy_ndimage.median_filter(data, footprint=kernel)

    kernel = kernel.astype(bool)

    func = filters[('median', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def median_filter_weigthed(data, kernel, mask=ParameterNotSpecified,
                           mode='ignore', ignore_nan=False):
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    kernel = kernel.astype(np.int_)

    func = filters[('wmedian', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def min_filter(data, kernel, mask=ParameterNotSpecified,
               mode='ignore', ignore_nan=False):
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    kernel = kernel.astype(bool)

    func = filters[('min', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def max_filter(data, kernel, mask=ParameterNotSpecified,
               mode='ignore', ignore_nan=False):
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    kernel = kernel.astype(bool)

    func = filters[('max', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def sum_filter(data, kernel, mask=ParameterNotSpecified,
               mode='ignore', ignore_nan=False):
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    func = filters[('sum', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def average_filter(data, kernel, mask=ParameterNotSpecified,
                   mode='ignore', ignore_nan=False):
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    func = filters[('mean', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)