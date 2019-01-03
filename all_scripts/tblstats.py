# -*- coding: UTF-8 -*-
"""
tblstats
===========

Script :   tblstats.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-11-23

Purpose :  Descriptive statistics for tables using numpy.

References:
-----------

`<https://github.com/numpy/numpy/blob/master/numpy/lib/nanfunctions.py>`_.

_replace_nan(a, val) -  mask = np.isnan(a) - to get the mask

>>> a = [1, 2, np.nan, 3, np.nan, 4]
>>> _, mask = _replace_nan(a, 0)  # for mean
>>> mask = array([False, False,  True, False,  True, False], dtype=bool)

"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ----

import sys
import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['freq',
           'summ',
           'skew_kurt',         # called by _calc_stats
           '_calc_stats',       # called by col_stats
           '_numeric_fields_',  # called by col_stats
           'col_stats',         # called by col_stats
           'group_stats',       # called by col_stats
           ]

if 'prn' not in locals().keys():
    try:
        from arraytools.frmts import prn
        # print("`prn` imported from arraytools")
    except:
        prn = print


def freq(a, flds=None, to_array=True):
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
    (4) reassemble the original columns and the new count data
    """
    if flds is None:
        return None
    if isinstance(flds, (str)):
        flds = [flds]
    a = a[flds]  # (1) slice
    idx = np.argsort(a, axis=0, order=flds)  # (2) sort
    a_sort = a[idx]
    uniq, counts = np.unique(a_sort, return_counts=True)  # (3) unique, count
    dt = uniq.dtype.descr + [('Count', '<i4')]
    fr = np.zeros_like(uniq, dtype=dt)
    names = fr.dtype.names
    vals = list(zip(*uniq)) + [counts.tolist()]  # (4) reassemble
    for i in range(len(names)):
        fr[names[i]] = vals[i]
    if to_array:
        return fr
    else:
        prn(fr)


def summ(a, cls_flds, uniq, sum_flds):
    """sum the input field

    `a` : array
        large array sliced by the classification fields
    `cls_fields` : fields
        fields to slice the array with
    `uniq` : string
        unique values to sum on
    `sum_flds` : string or list
        The fields to do the sum on
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
    a : array
        an array of float/double values where there are at least 3 non-nan
        numbers in each column.  This is not checked since this situation
        should never arise in real world data sets that have been checked.
    moment : string
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
    if mom == 'kurt':
        return kurt_m
    if mom == 'both':
        return skew_m, kurt_m

# ---- calculate field statistics section -----------------------------------
#
def _calc_stats(arr, axis=None, deci=4):
    """Calculate stats for an array of number types, with nodata (nan, None)
    in the column.

    Notes:
    -----
    see the args tuple for examples of nan functions

    >>> np.nansum(b, axis=0)   # by column
    >>> np.nansum(b, axis=1)   # by row
    >>> c_nan = np.count_nonzero(~np.isnan(b), axis=0) count nan if needed

    [1, 0][True]  # ax = [1, 0][colwise]  colwise= True
    """
    if (axis is None) and (len(arr.shape) == 1):
        ax = 0
    else:
        ax = axis
    #
    kind = arr.dtype.kind
    arr_dt = arr.dtype
    if kind == 'i':
        nulls = [np.iinfo(arr_dt).min, np.iinfo(arr_dt).max]
    elif kind == 'f':
        nulls = [np.nan, np.finfo(arr_dt).min, np.finfo(arr_dt).max]
    elif kind in ('U', 'S'):
        return None
    #
    nin = ~np.isin(arr, nulls)  # nin... Not In Nulls
    a = arr[nin]
    if len(arr.shape) > 1:
        a = a.reshape(arr.shape)
    mask = np.isnan(arr)
    N = len(a)
    cnt = np.sum(~mask, axis=ax, dtype=np.intp, keepdims=False)
    n_sum = np.nansum(a, axis=ax)
    n_min = np.nanmin(a, axis=ax)
    n_max = np.nanmax(a, axis=ax)
    n_mean = np.nanmean(a, axis=ax)
    n_med = np.nanmedian(a, axis=ax)
    n_std = np.nanstd(a, axis=ax)
    n_var = np.nanvar(a, axis=ax)
    sk, kurt = skew_kurt(a, avg=n_mean, var_x=n_var, std_x=n_std,
                         col=True, mom='both')
    s = [N, N-cnt, n_sum, n_min, n_max, n_mean, n_med, n_std, n_var, sk, kurt]
    s = [np.around(i, deci)  for i in s]
    return s


def _numeric_fields_(a, fields):
    """Determine numeric fields in a structured/recarray
    """
    num_flds = []
    dt_names = a.dtype.names
    dt_kind = a.dtype.kind
    if fields is None:
        if dt_names is None:
            if dt_kind not in ('i', 'f'):
                return None
        elif dt_kind in ['V']:
            num_flds = [i for i in dt_names if a[i].dtype.kind in ('i', 'f')]
        else:
            a = a.ravel()
    elif isinstance(fields, (str)):
        if a[fields].dtype.kind in ('i', 'f'):
            num_flds = fields
    else:
        num_flds = [i for i in fields if a[i].dtype.kind in ('i', 'f')]
    return num_flds


def col_stats(a, fields=None, deci=2):
    """Calculate statistics for a structured/recarray with or without specified
    fields.  Efforts have been made to check for all possible scenarios, but
    human intelligence should prevail when one decides what to throw at it.

    >>> a.dtype.names  # to return a list of field names
    >>> col_stats(a, fields='A')  # run with field 'A', must be integer/float
    >>> col_stats(a, fields=['A', 'B'])  # both fields checked

    Parameters:
    -----------
    a : array
        A structured/recarray
    fields : list, string or None
      - None,  checks all fields or assumes that the input array is a singleton
      - string, a single field name, if the column names are known
      - list,  a list of field names
    deci : integer
        an attempt to format floats with deci(mal) places

    Requires:
    ---------
    _numeric_fields_ : function
        returns the numeric fields in a structured/recarray
    _calc_stats : function
        performs the actual field calculations
    """
    s_lst = []
    num_flds = _numeric_fields_(a, fields)
    # ---- made it thus far
    if len(num_flds) == 0:
        num_flds = ['array']
        s_lst.append(_calc_stats(a.ravel(), axis=None, deci=deci))
    else:
        for fld in num_flds:
            s_lst.append(_calc_stats(a[fld], deci=deci))
    #
    dts = [('Statistic', 'U10')] + [(i, '<f8') for i in num_flds]
    col_names = np.array(['N (size)', 'n (nans)', 'sum', 'min', 'max', 'mean',
                          'median', 'std', 'var', 'skew', 'kurt'])
    z = np.zeros((len(col_names),), dtype=dts)
    z['Statistic'] = col_names
    for i in range(len(num_flds)):
        fld = num_flds[i]
        z[fld] = s_lst[i]
    return z


def group_stats(a, case_fld=None, num_flds=None, deci=2):
    """Group column statistics.

    Parameters:
    -----------
    a : structured/recarray
        Make sure that you know the field names in advance
    case_fld : string, list
        String,  summarized by the unique values in the case_fld.
        List, to further fine-tune the selection or crosstabulation
    num_flds : string, list
        You can limit the input fields accordingly, if you only need a few
        know numeric fields.

    Requires:
    ---------
    col_stats : function ... which requires
      : _numeric_fields_ : function
          returns the numeric fields in a structured/recarray
      : _calc_stats : function
          performs the actual field calculations

    """
    results = []
    uniq, counts = np.unique(a[case_fld], return_counts=True)
    n = len(uniq)
    for i in range(n):
        if counts[i] >= 3:
            u = uniq[i]
            sub = a[a[case_fld] == u]
            z = col_stats(sub, fields=num_flds, deci=deci)
            prn(z, title='a[{}] stats...'.format(u))
            results.append(z)
        else:
            print("\nToo few cases... ({}) for a[{}]...".format(counts[i], u))
    return results


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
