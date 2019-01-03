# -*- coding: UTF-8 -*-
"""
arr_moving
==========

Script :   arr_moving.py

Author :   Dan.Patterson@carleton.ca

Modified : 2016-07-31

Purpose :  Functions to import for use with numpy arrays.

Functions :
  public  -  block, stride_, deline, rolling_stats (_check required)

  private - _check, _pad, _demo

Notes :
  The following demonstration, shows how deline, stride, block
  and rolling_stats work.


**Rolling_stats example**

Stride the input array, 'a' using a 3x3 window

>>> c = stride(a, r_c=(3,3))
Main array... 'a'
ndim: 2 size: 54
shape: (6, 9)
[[ 0  1 ...,  7  8]
 [ 9 10 ..., 16 17]
 ...,
 [36 37 ..., 43 44]
 [45 46 ..., 52 53]]

>>> print(deline(c[0][:2]))
Main array...
ndim: 3 size: 18
shape: (2, 3, 3)
[[[ 0  1  2]
  [ 9 10 11]
  [18 19 20]]
a[1]....
 [[ 1  2  3]
  [10 11 12]
  [19 20 21]]]
----- big clip

>>> ax = tuple(np.arange(len(c.shape))[-2:]) # ax == (2, 3)
>>> c_m =np.mean(c, ax)
>>> print(deline(c_m))
Main array...
ndim: 2 size: 28
shape: (4, 7)
[[ 10.0  11.0 ...,  15.0  16.0]
 [ 19.0  20.0 ...,  24.0  25.0]
 [ 28.0  29.0 ...,  33.0  34.0]
 [ 37.0  38.0 ...,  42.0  43.0]]

Which is what we expect.  The trick is to produce the
stats on the last 2 entries in the array's shape.  If we do this
with a normal 2D array, like 'a', we get...


    np.mean(a,axis=(0,1)) == 26.5


which for the whole 6*9 array rather than moving 3*3 window slices
through it.  Striding or blocking a 2D array, results in a 4D
array, statistics are calculate on the last 2 dimensions (2,3)


**Block stats example**

Following the same procedure above, the results are

>>> d = block(a, r_c=(3,3))
 array([[[[ 0,  1,  2],
          [ 9, 10, 11],
          [18, 19, 20]],
        .... snip .....
        [[33, 34, 35],
         [42, 43, 44],
         [51, 52, 53]]]])
>>> ax = tuple(np.arange(len(c.shape))[-2:]) which is (2,3) for 4D
>>> d_m =np.mean(d, ax)
>>> print(deline(d_m))
 Main array...
 ndim: 2 size: 6
 shape: (2, 3)
 [[ 10.0  13.0  16.0]
  [ 37.0  40.0  43.0]]


**Masked arrays**

>>> m = np.where((a>19) & (a<27),1,0)
>>> a_msk = np.ma.MaskedArray(a, mask=m, dtype='float')  (fill value 1e 200)
>>> a_msk = np.ma.MaskedArray(a, mask=m, dtype='float',fill_value=np.nan)

last option useful for simplifying nodata values

>>> a_msk
masked_array(data =
 [[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
  [9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0]
  [18.0 19.0 -- -- -- -- -- -- --]
  [27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0]
  [36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0]
  [45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0]],
          mask =
 [[0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0]
  [0 0 1 1 1 1 1 1 1]
  [0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0]],
        fill_value = nan)


>>> np.mean(a)        => 26.5 ! nan values not accounted for, hence wrong
>>> np.mean(a_msk)    => 27.021276595744681  accounted for
>>> np.ma.mean(a_msk) => 27.021276595744681


References:
-----------


"""

# ---- imports, formats, constants ----

import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
from textwrap import dedent
from arraytools.frmts import deline

__all__ = ['_check', 'block', 'stride_', 'rolling_stats']
__outside__ = ['as_strided', 'dedent', 'deline']

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)

script = sys.argv[0]


# ---- functions ----

def _check(a, r_c):
    """Performs the array checks necessary for stride and block.

    `a` : array-like
        Array or list.
    `r_c` : tuple/list/array of rows x cols.
        Attempts will be made to produce a shape at least (1*c).  If a scalar
        is given, the minimum shape will be (1*r) for 1D array or (1*c) for
        2D array if r<c.  Be aware
    """
    if isinstance(r_c, (int, float)):
        r_c = (1, int(r_c))
    r, c = r_c
    a = np.atleast_2d(a)
    shp = a.shape
    r, c = r_c = (min(r, a.shape[0]), min(c, shp[1]))
    a = np.ascontiguousarray(a)
    return a, shp, r, c, tuple(r_c)


def _pad(a, nan_edge=False):
    """Pad a sliding array to allow for stats"""
    if nan_edge:
        a = np.pad(a, pad_width=(1, 2), mode="constant",
                   constant_values=(np.NaN, np.NaN))
    else:
        a = np.pad(a, pad_width=(1, 1), mode="reflect")
    return a


def stride_(a, r_c=(3, 3)):
    """Provide a 2D sliding/moving view of an array.
    There is no edge correction for outputs.

    Requires:
    --------
    `a` : array-like
        Array or list, usually a 2D array.  Assumes the rows is >=1, it is
        corrected as is the number of columns.
    `r_c` : tuple/list/array of rows x cols.
        Attempts will be made to produce a shape at least (1*c).  If a scalar
        is given, the minimum shape will be (1*r) for 1D array or (1*c)
        for 2D array if r<c.  Be aware

    See also:
    ---------
        A more detailed version of `stride` is available in `tools.py`
    """
    a, shp, r, c, r_c = _check(a, r_c)
    shp = (a.shape[0] - r + 1, a.shape[1] - c + 1) + r_c
    strd = a.strides * 2
    a_s = (as_strided(a, shape=shp, strides=strd)).squeeze()
    return a_s


def block(a, r_c=(3, 3)):
    """See _check and/or stride for documentation.  This function  moves in
    increments of the block size, rather than sliding by one row and column.
    """
    a, shp, r, c, r_c = _check(a, r_c)
    shp = (a.shape[0]//r, a.shape[1]//c) + r_c
    strd = (r*a.strides[0], c*a.strides[1]) + a.strides
    a_b = as_strided(a, shape=shp, strides=strd).squeeze()
    return a_b


def rolling_stats(a, no_null=True, prn=True):
    """Statistics on the last two dimensions of an array.

    Requires:
    --------
    `a` : array
        2D array
    `no_null` : boolean
        Whether to use masked values (nan) or not.
    `prn` : boolean
        To print the results or return the values.

    Returns:
    -------

    The results return an array of 4 dimensions representing the original
    array size and block size

    eg.  original = 6x6 array   block=3x3 ...breaking the array into 4 chunks
    """
    a = np.asarray(a)
    a = np.atleast_2d(a)
    ax = None
    if a.ndim > 1:
        ax = tuple(np.arange(len(a.shape))[-2:])
    if no_null:
        a_min = a.min(axis=ax)
        a_max = a.max(axis=ax)
        a_mean = a.mean(axis=ax)
        a_sum = a.sum(axis=ax)
        a_std = a.std(axis=ax)
        a_var = a.var(axis=ax)
        a_ptp = a_max - a_min
    else:
        a_min = np.nanmin(a, axis=(ax))
        a_max = np.nanmax(a, axis=(ax))
        a_mean = np.nanmean(a, axis=(ax))
        a_sum = np.nansum(a, axis=(ax))
        a_std = np.nanstd(a, axis=(ax))
        a_var = np.nanvar(a, axis=(ax))
        a_ptp = a_max - a_min
    if prn:
        frmt = "Minimum...\n{}\nMaximum...\n{}\nMean...\n{}\n" +\
               "Sum...\n{}\nStd...\n{}\nVar...\n{}\nRange...\n{}"
        frmt = dedent(frmt)
        args = [a_min, a_max, a_mean, a_sum, a_std, a_var, a_ptp]
        print(frmt.format(*args))
    else:
        return a_min, a_max, a_mean, a_sum, a_std, a_var, a_ptp


# -----------------------------------
def _demo():
    """
    :Run demo of block, for a 2D array which yields a 3D array
    :Run a stride of a 2D array which yields a 4D array
    """
    r = 6         # array rows
    c = 9         # array columns
    r_c = (3, 3)  # moving/block window size
    a = np.arange(r*c).reshape(r, c)
    b = block(a)
    c = stride_(a, r_c)
#    print(deline(a))
#    print(deline(b))
#    print(deline(c))
    frmt = """
    Rolling stats for 'a' using a 3x3 rolling window
    :array a...
    :  ndim {}  size {}
    :  print(deline(c[0][:2])
    {}
    :etc......\n
    :rolling mean:
    :  c = stride(a, r_c=(3,3))
    :  c.shape  # (4, 7, 3, 3)
    :  ax = tuple(np.arange(len(c.shape))[-2:]) #(0,1,2,3) => (2,3)
    :  c_m =np.mean(c, ax)
    : ==> {}
    """
    c_m = np.mean(c, axis=(2, 3))
    as0 = deline(c[0, :2])
    as1 = deline(c_m)
    args = [c.ndim, c.size, as0, as1]
    print(dedent(frmt).format(*args))
    return a, b, c


# ----------------------
if __name__ == "__main__":
    """   """
#    print("Script... {}".format(script))
#    a, b, c = _demo()
