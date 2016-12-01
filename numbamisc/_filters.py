from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

if os.environ.get('READTHEDOCS') != 'True':
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


class ParameterNotSpecified(object):
    def __str__(self):
        return "'unspecified'"

    __repr__ = __str__

ParameterNotSpecified = ParameterNotSpecified()


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
                  mode='ignore', ignore_nan=True):
    """Median filter ignoring masked and NaN values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to filter.

    kernel : :class:`int`, :class:`tuple`, :class:`~numpy.ndarray`, \
:class:`~astropy.convolution.Kernel`
        See main documentation for explanation.
        The ``kernel`` is cast to ``dtype=bool`` before filtering.

    mask : :class:`~numpy.ndarray`, optional
        Mask for the filter. If given an implicit mask by ``data`` is ignored.

    mode : string, optional
        How to treat values outside the ``data``.
        Default is ``ignore``.

    ignore_nan : bool, optional
        Also ignore ``NaN`` values.
        Default is ``True``.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The filtered array.

    mask : :class:`~numpy.ndarray`
        The mask of the filtered array.

    See also
    --------
    scipy.ndimage.median_filter : the same without mask support.

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

    Explictly setting kernel elements to zero excludes those elements for the
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
                           mode='ignore', ignore_nan=True):
    """Weighted median filter ignoring masked and NaN values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to filter.

    kernel : :class:`int`, :class:`tuple`, :class:`~numpy.ndarray`, \
:class:`~astropy.convolution.Kernel`
        See main documentation for explanation.
        The ``kernel`` is cast to ``dtype=np.int_`` before filtering.

    mask : :class:`~numpy.ndarray`, optional
        Mask for the filter. If given an implicit mask by ``data`` is ignored.

    mode : string, optional
        How to treat values outside the ``data``.
        Default is ``ignore``.

    ignore_nan : bool, optional
        Also ignore ``NaN`` values.
        Default is ``True``.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The filtered array.

    mask : :class:`~numpy.ndarray`
        The mask of the filtered array.
    """
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    kernel = kernel.astype(np.int_)

    func = filters[('wmedian', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def min_filter(data, kernel, mask=ParameterNotSpecified,
               mode='ignore', ignore_nan=True):
    """Minimum filter ignoring masked and NaN values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to filter.

    kernel : :class:`int`, :class:`tuple`, :class:`~numpy.ndarray`, \
:class:`~astropy.convolution.Kernel`
        See main documentation for explanation.
        The ``kernel`` is cast to ``dtype=bool`` before filtering.

    mask : :class:`~numpy.ndarray`, optional
        Mask for the filter. If given an implicit mask by ``data`` is ignored.

    mode : string, optional
        How to treat values outside the ``data``.
        Default is ``ignore``.

    ignore_nan : bool, optional
        Also ignore ``NaN`` values.
        Default is ``True``.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The median filtered array.

    mask : :class:`~numpy.ndarray`
        The mask of the filtered array.

    See also
    --------
    scipy.ndimage.minimum_filter : the same without mask support.
    """
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    kernel = kernel.astype(bool)

    func = filters[('min', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def max_filter(data, kernel, mask=ParameterNotSpecified,
               mode='ignore', ignore_nan=True):
    """Maximum filter ignoring masked and NaN values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to filter.

    kernel : :class:`int`, :class:`tuple`, :class:`~numpy.ndarray`, \
:class:`~astropy.convolution.Kernel`
        See main documentation for explanation.
        The ``kernel`` is cast to ``dtype=bool`` before filtering.

    mask : :class:`~numpy.ndarray`, optional
        Mask for the filter. If given an implicit mask by ``data`` is ignored.

    mode : string, optional
        How to treat values outside the ``data``.
        Default is ``ignore``.

    ignore_nan : bool, optional
        Also ignore ``NaN`` values.
        Default is ``True``.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The median filtered array.

    mask : :class:`~numpy.ndarray`
        The mask of the filtered array.

    See also
    --------
    scipy.ndimage.maximum_filter : the same without mask support.
    """
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    kernel = kernel.astype(bool)

    func = filters[('max', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def sum_filter(data, kernel, mask=ParameterNotSpecified,
               mode='ignore', ignore_nan=True):
    """Summation filter ignoring masked and NaN values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to filter.

    kernel : :class:`int`, :class:`tuple`, :class:`~numpy.ndarray`, \
:class:`~astropy.convolution.Kernel`
        See main documentation for explanation.
        The kernel must not contain mixed signed values. Either all values
        must be positive or all negative.

    mask : :class:`~numpy.ndarray`, optional
        Mask for the filter. If given an implicit mask by ``data`` is ignored.

    mode : string, optional
        How to treat values outside the ``data``.
        Default is ``ignore``.

    ignore_nan : bool, optional
        Also ignore ``NaN`` values.
        Default is ``True``.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The median filtered array.

    mask : :class:`~numpy.ndarray`
        The mask of the filtered array.

    See also
    --------
    scipy.ndimage.convolve : the same without mask support.
    """
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    func = filters[('sum', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)


def average_filter(data, kernel, mask=ParameterNotSpecified,
                   mode='ignore', ignore_nan=True):
    """Averaging filter ignoring masked and NaN values.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, \
:class:`~astropy.nddata.NDData`
        The data to filter.

    kernel : :class:`int`, :class:`tuple`, :class:`~numpy.ndarray`, \
:class:`~astropy.convolution.Kernel`
        See main documentation for explanation.
        The kernel must not contain mixed signed values. Either all values
        must be positive or all negative.

    mask : :class:`~numpy.ndarray`, optional
        Mask for the filter. If given an implicit mask by ``data`` is ignored.

    mode : string, optional
        How to treat values outside the ``data``.
        Default is ``ignore``.

    ignore_nan : bool, optional
        Also ignore ``NaN`` values.
        Default is ``True``.

    Returns
    -------
    filtered : :class:`~numpy.ndarray`
        The median filtered array.

    mask : :class:`~numpy.ndarray`
        The mask of the filtered array.
    """
    data, kernel, mask = _process_input(data, kernel, mask)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    func = filters[('mean', data.ndim, mode, ignore_nan)]

    return func(data, kernel, mask)
