import numpy as np

from jinja2 import Template
from numba import njit
from scipy.ndimage import median_filter as scipy_median_filter

__all__ = ['median_filter']


ParameterNotSpecified = object()


class MedianFilterTemplate(object):
    """Class to create and return the appropriate median_filter templates."""
    _template = Template("""
def median(image, kernel, mask):
    {% for i in range(0, dimensions) %}
    n{{i}} = image.shape[{{i}}]
    nk{{i}} = kernel.shape[{{i}}]
    wk{{i}} = nk{{i}} // 2
    {% endfor %}
    result = np.zeros(image.shape, dtype=np.float64)
    tmp = np.zeros(kernel.size, dtype=image.dtype)
    {% for i in range(0, dimensions) %}
    {{' ' * i * 4}}for i{{i}} in range(n{{i}}):
    {{' ' * i * 4}}    n{{i}}min = max(i{{i}} - wk{{i}}, 0)
    {{' ' * i * 4}}    n{{i}}max = min(i{{i}} + wk{{i}} + 1, n{{i}})
    {% endfor %}
    {{' ' * dimensions * 4}}elements = 0
    {% for i in range(0, dimensions) %}
    {{' ' * dimensions * 4}}{{' ' * i * 4}}for ii{{i}} in range(n{{i}}min, n{{i}}max):
    {{' ' * dimensions * 4}}{{' ' * i * 4}}    iii{{i}} = wk{{i}} + ii{{i}} - i{{i}}
    {% endfor %}
    {{' ' * dimensions * 8}}if (not mask[{% for i in range(0, dimensions-1) %}ii{{i}}, {% endfor %}ii{{dimensions-1}}] and kernel[{% for i in range(0, dimensions-1) %}iii{{i}}, {% endfor %}iii{{dimensions-1}}]):
    {{' ' * dimensions * 8}}    tmp[elements] = image[{% for i in range(0, dimensions-1) %}ii{{i}}, {% endfor %}ii{{dimensions-1}}]
    {{' ' * dimensions * 8}}    elements += 1

    {{' ' * dimensions * 4}}if elements > 0:
    {{' ' * dimensions * 4}}    result[{% for i in range(0, dimensions-1) %}i{{i}}, {% endfor %}i{{dimensions-1}}] = np.median(tmp[:elements])
    {{' ' * dimensions * 4}}else:
    {{' ' * dimensions * 4}}    result[{% for i in range(0, dimensions-1) %}i{{i}}, {% endfor %}i{{dimensions-1}}] = np.nan
    return result
""")

    _funcs = {}

    def __init__(self):
        pass

    def get_template(self, dimensions):
        if dimensions not in self._funcs:
            exec(self._template.render(dimensions=dimensions), globals())
            # median function is created in the template execution.
            self._funcs[dimensions] = njit(median)
        return self._funcs[dimensions]


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


def _process(data, kernel, mask):
    """Checks prerequisites and loads the appropriate function.
    """

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

    data = _convert_to_native_bytorder(data)

    # It is possible that the kernel is an astropy kernel, in that case it has
    # an attribute "array" and we use that one:
    kernel = _convert_to_native_bytorder(getattr(kernel, 'array', kernel))

    # Only in case no explicit mask was given use the one extracted from the
    # data.
    if mask is ParameterNotSpecified:
        mask = mask2

    # In case we have no mask (None) simply use scipy.median_filter
    if mask is None:
        return scipy_median_filter(data, footprint=kernel)

    # Check if the shape is the same. There might be cases where the
    # array contained a mask attribute but the mask has a different shape
    # than the data!
    else:
        mask = np.asarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')

    # Evaluate how many dimensions the array has, this is needed to find the
    # appropriate convolution or interpolation function.
    ndim = data.ndim

    # kernel must have the same number of dimensions
    if kernel.ndim != ndim:
        raise ValueError('data and kernel must have the same number of '
                         'dimensions.')

    # the kernel also needs to be odd in every dimension.
    if any(i % 2 == 0 for i in kernel.shape):
        raise ValueError('kernel must have an odd shape in each dimension.')

    func = MedianFilterTemplate().get_template(ndim)

    return func(data, kernel, mask)


def median_filter(data, kernel, mask=ParameterNotSpecified):
    """Median based convolution of some data by ignoring masked values.

    .. note::
        Requires ``numba``.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ma.MaskedArray`, :class:`~astropy.nddata.NDData`
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
    This allows using :class:`~numpy.ma.MaskedArray` objects as ``data`` parameter.

    If an explicit ``mask`` is given (even if it is ``None``) an implicit
    mask is ignored.

    No border handling is possible, if the kernel extends beyond the image
    these *outside* values are treated as if they were masked.

    Examples
    --------
    A simple example::

        >>> from numbamisc import median_filter
        >>> import numpy as np

        >>> data = np.ma.array([1,1000,2,1], mask=[0, 1, 0, 0])
        >>> median_filter(data, [1,1,1])
        array([ 1. ,  1.5,  1.5,  1.5])

    Support for arbitarly dimensional arrays and masks is also implemented::

        >>> data = np.arange(9).reshape(3, 3)
        >>> data[1, 1] = 100
        >>> mask = np.zeros((3, 3), dtype=bool)
        >>> mask[1, 1] = 1
        >>> median_filter(data, np.ones((3,3)), mask)
        array([[ 1.,  2.,  2.],
               [ 3.,  4.,  5.],
               [ 6.,  6.,  7.]])

    And another example::

        >>> data = np.arange(27).reshape(3, 3, 3)
        >>> data[0, 0, 0] = 10000
        >>> mask = np.zeros((3, 3, 3))
        >>> mask[0, 0, 0] = 1
        >>> median_filter(data, np.ones((3, 3, 3)), mask)
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
        >>> median_filter(data, [1, 0, 0])
        array([ nan,   1.,  nan,   2.])

    Here only the left element is used for the convolution. For the first
    element the left one is outside the data and for the third element the
    convolution element is masked so both of them result in ``NaN``.
    """
    return _process(data, kernel, mask)
