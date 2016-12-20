from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

if os.environ.get('READTHEDOCS') != 'True':
    import jinja2


__all__ = ['generate']


def stripblanklines(tmp):
    tmp = tmp.split('\n')
    tmp = (i for i in tmp if i.strip())
    return '\n'.join(tmp)


_init = """
import math
import numba as nb
import numpy as np


filters = {}


@nb.njit(nogil=True, cache=True)
def min_ii_ignore(i, nkl, n):
    return max(i - nkl, 0)


@nb.njit(nogil=True, cache=True)
def min_ii_default(i, nkl, n):
    return i - nkl


@nb.njit(nogil=True, cache=True)
def max_ii_ignore(i, nkh, n):
    return min(i + nkh + 1, n)


@nb.njit(nogil=True, cache=True)
def max_ii_default(i, nkh, n):
    return i + nkh + 1


@nb.njit(nogil=True, cache=True)
def img_idx_nearest(ii, n):
    if ii < 0:
        ii = 0
    elif ii >= n:
        ii = n - 1
    return ii


@nb.njit(nogil=True, cache=True)
def img_idx_wrap(ii, n):
    return ii % n


@nb.njit(nogil=True, cache=True)
def img_idx_mirror(ii, n):
    while ii < 0 or ii >= n:
        if ii < 0:
            ii = -ii
        elif ii >= n:
            ii = 2 * n - 2 - ii
    return ii


@nb.njit(nogil=True, cache=True)
def img_idx_reflect(ii, n):
    while ii < 0 or ii >= n:
        if ii < 0:
            ii = -ii - 1
        elif ii >= n:
            ii = 2 * n - 1 - ii
    return ii


@nb.njit
def _partition(A, low, high, k):
    '''Function taken from numba source code and modified.

    BSD licensed see numba license.
    '''
    if high - low < 6:
        insertion_sort(A, low, high)
        return k

    mid = (low + high) >> 1
    if A[mid] < A[low]:
        A[low], A[mid] = A[mid], A[low]
    if A[high] < A[mid]:
        A[high], A[mid] = A[mid], A[high]
        if A[mid] < A[low]:
            A[low], A[mid] = A[mid], A[low]
    pivot = A[mid]

    A[high], A[mid] = A[mid], A[high]

    i = low
    for j in range(low, high):
        if A[j] <= pivot:
            A[i], A[j] = A[j], A[i]
            i += 1

    A[i], A[high] = A[high], A[i]
    return i


@nb.njit
def _select(arry, k, low, high):
    '''Function taken from numba source code and modified.

    BSD licensed see numba license.
    '''
    i = _partition(arry, low, high, k)
    while i != k:
        if i < k:
            low = i + 1
            i = _partition(arry, low, high, k)
        else:
            high = i - 1
            i = _partition(arry, low, high, k)
    return arry[k]


@nb.njit
def _select_two(arry, k, low, high):
    '''Function taken from numba source code and modified.

    BSD licensed see numba license.
    '''
    while True:
        i = _partition(arry, low, high, k)
        if i < k:
            low = i + 1
        elif i > k + 1:
            high = i - 1
        elif i == k:
            _select(arry, k + 1, i + 1, high)
            break
        else:  # i == k + 1
            _select(arry, k, low, i - 1)
            break

    return arry[k], arry[k + 1]


@nb.njit
def median(temp_arry):
    '''Function taken from numba source code and modified.

    BSD licensed see numba license.

    The original numba_median function copied the array and did not fallback
    to insertion sort for short arrays. Therefore this slightly optimized
    copy.
    '''
    n = temp_arry.size
    low = 0
    high = n - 1
    half = n >> 1
    if n & 1 == 0:
        a, b = _select_two(temp_arry, half - 1, low, high)
        return (a + b) / 2.
    else:
        return _select(temp_arry, half, low, high)


@nb.njit
def insertion_sort(array, low, high):
    for i in range(low + 1, high + 1):
        j = i
        while j > low and array[j] < array[j-1]:
            array[j], array[j-1] = array[j-1], array[j]
            j -= 1
"""

_convolutiontemplate = """
@nb.njit(nogil=True, cache=True)
def filter_{{mode}}_n{{ndim}}_{{out}}{{nan}}(image, kernel, mask):
    {% for i in range(ndim) %}
    n{{i}} = image.shape[{{i}}]
    nkh{{i}} = kernel.shape[{{i}}] // 2
    nkl{{i}} = nkh{{i}}
    if kernel.shape[{{i}}] % 2 == 0:
        nkh{{i}} -= 1
    {% endfor %}
    resdata = np.zeros(image.shape, dtype=np.float64)
    resmask = np.zeros(image.shape, dtype=np.bool_)

    {% if mode == 'median' %}
    tmp = np.zeros(np.sum(kernel), dtype=image.dtype)
    {% elif mode == 'wmedian' %}
    tmp = np.zeros(np.sum(kernel), dtype=image.dtype)
    {% elif mode == 'sum'%}
    kernelsum = np.sum(kernel)
    {% endif %}

    {% for i in range(ndim) %}
    {% if True             %}{{ ' '*4*i       }}for i{{i}} in range(n{{i}}):{% endif %}
    {% if out == 'ignore'  %}{{ ' '*4*i       }}    n{{i}}min = min_ii_ignore(i{{i}}, nkl{{i}}, n{{i}})
    {% else                %}{{ ' '*4*i       }}    n{{i}}min = min_ii_default(i{{i}}, nkl{{i}}, n{{i}})
    {% endif %}
    {% if out == 'ignore'  %}{{ ' '*4*i       }}    n{{i}}max = max_ii_ignore(i{{i}}, nkh{{i}}, n{{i}})
    {% else                %}{{ ' '*4*i       }}    n{{i}}max = max_ii_default(i{{i}}, nkh{{i}}, n{{i}})
    {% endif %}
    {% endfor %}

    {% if mode == 'min'     %}{{ ' '*4*ndim    }}tmpmin = None{% endif %}
    {% if mode == 'max'     %}{{ ' '*4*ndim    }}tmpmax = None{% endif %}
    {% if mode == 'median'  %}{{ ' '*4*ndim    }}elements = 0{% endif %}
    {% if mode == 'wmedian' %}{{ ' '*4*ndim    }}elements = 0{% endif %}
    {% if mode == 'sum'     %}{{ ' '*4*ndim    }}acc = 0.{% endif %}
    {% if mode == 'sum'     %}{{ ' '*4*ndim    }}div = 0.{% endif %}
    {% if mode == 'mean'    %}{{ ' '*4*ndim    }}acc = 0.{% endif %}
    {% if mode == 'mean'    %}{{ ' '*4*ndim    }}div = 0.{% endif %}

    {% for i in range(ndim) %}
    {% if True             %}{{ ' '*4*(i+ndim) }}for ii{{i}} in range(n{{i}}min, n{{i}}max):{% endif %}
    {% if True             %}{{ ' '*4*(i+ndim) }}    iii{{i}} = nkl{{i}} + ii{{i}} - i{{i}}{% endif %}
    {% if out == 'nearest' %}{{ ' '*4*(i+ndim) }}    ii{{i}} = img_idx_nearest(ii{{i}}, n{{i}}){% endif %}
    {% if out == 'wrap'    %}{{ ' '*4*(i+ndim) }}    ii{{i}} = img_idx_wrap(ii{{i}}, n{{i}}){% endif %}
    {% if out == 'reflect' %}{{ ' '*4*(i+ndim) }}    ii{{i}} = img_idx_reflect(ii{{i}}, n{{i}}){% endif %}
    {% if out == 'mirror'  %}{{ ' '*4*(i+ndim) }}    ii{{i}} = img_idx_mirror(ii{{i}}, n{{i}}){% endif %}
    {% endfor %}

    {{ ' '*8*ndim }}imgitem = image[{% for i in range(ndim-1) %}ii{{i}}, {% endfor %}ii{{ndim-1}}]
    {{ ' '*8*ndim }}maskitem = mask[{% for i in range(ndim-1) %}ii{{i}}, {% endfor %}ii{{ndim-1}}]
    {{ ' '*8*ndim }}kernitem = kernel[{% for i in range(ndim-1) %}iii{{i}}, {% endfor %}iii{{ndim-1}}]
    {{ ' '*8*ndim }}if kernitem and not maskitem{% if nan %} and not math.isnan(imgitem){% endif %}:
    {% if mode == 'median' %}
    {{ ' '*8*ndim }}    tmp[elements] = imgitem
    {{ ' '*8*ndim }}    elements += 1
    {% elif mode == 'min' %}
    {{ ' '*8*ndim }}    if tmpmin is None:
    {{ ' '*8*ndim }}        tmpmin = imgitem
    {{ ' '*8*ndim }}    elif imgitem < tmpmin:
    {{ ' '*8*ndim }}        tmpmin = imgitem
    {% elif mode == 'max' %}
    {{ ' '*8*ndim }}    if tmpmax is None:
    {{ ' '*8*ndim }}        tmpmax = imgitem
    {{ ' '*8*ndim }}    elif imgitem > tmpmax:
    {{ ' '*8*ndim }}        tmpmax = imgitem
    {% elif mode == 'wmedian' %}
    {{ ' '*8*ndim }}    for _ in range(kernitem):
    {{ ' '*8*ndim }}        tmp[elements] = imgitem
    {{ ' '*8*ndim }}        elements += 1
    {% elif mode == 'sum' %}
    {{ ' '*8*ndim }}    acc += imgitem * kernitem
    {{ ' '*8*ndim }}    div += kernitem
    {% elif mode == 'mean' %}
    {{ ' '*8*ndim }}    acc += imgitem * kernitem
    {{ ' '*8*ndim }}    div += kernitem
    {% endif %}

    {% if mode == 'median' %}
    {{ ' '*4*ndim }}if elements != 0:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = median(tmp[:elements])
    {% elif mode == 'min' %}
    {{ ' '*4*ndim }}if tmpmin is not None:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = tmpmin
    {% elif mode == 'max' %}
    {{ ' '*4*ndim }}if tmpmax is not None:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = tmpmax
    {% elif mode == 'wmedian' %}
    {{ ' '*4*ndim }}if elements != 0:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = median(tmp[:elements])
    {% elif mode == 'sum' %}
    {{ ' '*4*ndim }}if div != 0.:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = acc / div * kernelsum
    {% elif mode == 'mean' %}
    {{ ' '*4*ndim }}if div != 0.:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = acc / div
    {% endif %}
    {{ ' '*4*ndim }}else:
    {{ ' '*4*ndim }}    resdata[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = np.nan
    {{ ' '*4*ndim }}    resmask[{% for i in range(0, ndim-1) %}i{{i}}, {% endfor %}i{{ndim-1}}] = True

    return resdata, resmask

filters[('{{mode}}', {{ndim}}, '{{out}}', bool('{{nan}}'))] = filter_{{mode}}_n{{ndim}}_{{out}}{{nan}}
"""


def generate(maxndim):
    _tmpl = jinja2.Template(_convolutiontemplate)
    _ndim = range(1, maxndim + 1)
    _modes = ['median', 'wmedian', 'sum', 'mean', 'min', 'max']
    _out = ['ignore', 'nearest', 'wrap', 'reflect', 'mirror']
    _nan = ['_ignore_nan', '']

    _funcs = [stripblanklines(_tmpl.render(ndim=ndim, mode=mode,
                                           out=out, nan=nan))
              for mode in _modes
              for ndim in _ndim
              for out in _out
              for nan in _nan]
    _funcs.insert(0, _init)
    return '\n\n\n'.join(_funcs) + '\n'
