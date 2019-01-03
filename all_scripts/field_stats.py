# -*- coding: UTF-8 -*-
"""
field_stats
===========

Script :   field_stats.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-03-21

Purpose :  Descriptive statistics for tables using numpy.

References:
[1]
https://github.com/numpy/numpy/blob/master/numpy/lib/nanfunctions.py

  _replace_nan(a, val) -  mask = np.isnan(a) - to get the mask

>>> a = [1, 2, np.nan, 3, np.nan, 4]
>>> _, mask = _replace_nan(a, 0)  # for mean
>>> mask = array([False, False,  True, False,  True, False], dtype=bool)

: ---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


if 'prn' not in locals().keys():
    try:
        from arraytools.frmts import prn
    except:
        prn = print


def freq(a, flds=None):
    """Frequency and crosstabulation

    `a` : array
       input structured array

    `flds` : string or list
       fields/columns to use in the analysis

    Notes
    -----
    (1) slice the input array by the classification fields
    (2) sort the sliced array using the flds as sorting keys
    (3) use unique on the sorted array to return the results
    (4) a quick histogram to get the counts until numpy 1.12 can be used
        then ship the results back.  only uni and vals is needed. The
        rest is for testing and future work.
    """
    if flds is None:
        return None
    elif isinstance(flds, (str)):
        flds = [flds]
    a = a[flds]  # (1)
    idx = np.argsort(a, axis=0, order=flds)  # (2)
    a_sort = a[idx]
    uniq, counts = np.unique(a_sort, return_counts=True)  # (3)
    dt = uniq.dtype.descr + [('Count', '<i4')]
    fr = np.zeros_like(uniq, dtype=dt)
    names = fr.dtype.names
    vals = uniq, counts
    for i in range(len(names)):
        fr[names[i]] =  vals[i]
    return fr


def summ(a, cls_flds, uniq, sum_flds):
    """sum the input field
    : a is the large array sliced by the classification fields
    : uniq - unique classes
    :
    """
    to_sum = a[cls_flds]
    out_sum = []
    for cl in uniq:
        rows = a[to_sum == cl]
        out_sum.append(np.nansum(rows[sum_flds]))  # use nansum
    return out_sum

# ---- skewness and kurtosis section -----------------------------------------

def skew_kurt(a, avg, var_x, std_x, col=True, mom='both'):
    """Momental and unbiased skewness

    Emulates the nan functions approach to calculating these parameters
    when data contains nan values.

    Requires
    ---------
    a :
        an array of float/double values where there are at least 3 non-nan
        numbers in each column.  This is not checked since this situation
        should never arise in real world data sets that have been checked.
    moment :
        both, skew or kurt  to return the moments

    Notes:
    -----
        >>> a= np.arange(16.).reshape(4,4)
        >>> mask = [0, 5, 10, 15]
        >>> masked_array = np.where(a == mask, np.nan, a)
    """
#    a, mask = _replace_nan(a, 0.)  # produce a masked of the nan values
    if len(a.shape) == 1:
        ax = 0
    else:
        ax = [1, 0][col]
#    # ---- mean section ----
    mask = np.isnan(a)
    cnt = np.sum(~mask, axis=ax, dtype=np.intp, keepdims=False)
    diff = a - avg
    sqrd = diff * diff
    cubed = sqrd * diff
    fourP = sqrd * sqrd
    x_3 = np.nansum(cubed, axis=ax)
    x_4 = np.nansum(fourP, axis=ax)
    skew_m = x_3 / (cnt * (std_x**3))
    kurt_m = x_4 / (cnt * (var_x * var_x))
    # skew_u = skew_m*((cnt**2)/((cnt-1)*(cnt-2)))  # could add if needed
    if mom == 'skew':
        return skew_m
    elif mom == 'kurt':
        return kurt_m
    elif mom == 'both':
        return skew_m, kurt_m


def cal_stats(a):
    """Calculate stats for an array of double types, with nodata (nan, None)
    in the column.

    Notes
    -----
    see the args tuple for examples of nan functions::

        >>> np.nansum(b, axis=0)   # by column
        >>> np.nansum(b, axis=1)   # by row
        >>> c_nan = np.count_nonzero(~np.isnan(b), axis=0) count nan if needed
    """
    if len(a.shape) == 1:
        ax = 0
    else:
        ax = [1, 0][True]  # ax = [1, 0][colwise]  colwise= True
    mask = np.isnan(a)
    n = len(a)
    cnt = np.sum(~mask, axis=ax, dtype=np.intp, keepdims=False)
    n_sum = np.nansum(a, axis=0)
    n_min = np.nanmin(a, axis=0)
    n_max = np.nanmax(a, axis=0)
    n_mean = np.nanmean(a, axis=0)
    n_med = np.nanmedian(a, axis=0)
    n_std = np.nanstd(a, axis=0)
    n_var = np.nanvar(a, axis=0)
    col_names = ['N', 'n', 'sum', 'min', 'max', 'mean', 'median',
                 'std', 'var', 'skew', 'kurt']
    sk, kurt = skew_kurt(a, avg=n_mean, var_x=n_var, std_x=n_std,
                         col=True, mom='both')
    args = [n, cnt, n_sum, n_min, n_max, n_mean, n_med, n_std, n_var, sk, kurt]
    z = list(zip(col_names, args))
    s = "".join(["\n{:<6} {}".format(*i) for i in z])
    return s


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
