# -*- coding: utf-8 -*-
"""
script name here
=======

Script :   template.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-08-25

Purpose :  Tools for working with tabular data in the Geo class

Notes:

References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np
from npgeom import npg_io
from npg_io import prn_tbl

# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import repack_fields
# from numpy.lib.recfunctions import _keep_fields
from numpy.lib.recfunctions import append_fields

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(
        edgeitems=10, linewidth=80, precision=2, suppress=True, threshold=100,
        formatter=ft
        )
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['crosstab_tbl', 'crosstab_rc', 'crosstab_array', 'col_stats',
           'group_stats']


# ==== Crosstabulation tools =================================================
# ---- fancy print/string formatter for crosstabulation and pivot
def _prn(r, c, a, stat_name='Count'):
    """fancy print formatting.
    """
    r = r.tolist()
    r.append(stat_name)
    c = c.tolist()
    c.append(stat_name)
    r_sze = max(max([len(str(i)) for i in r]), 8)
    c_sze = [max(len(str(i)), 5) for i in c]
    f_0 = '{{!s:<{}}} '.format(r_sze)
    f_1 = ('{{!s:>{}}} '*len(c)).format(*c_sze)
    frmt = f_0 + f_1
    hdr = 'Result' + '_'*(r_sze-7)
    txt = [frmt.format(hdr, *c)]
    txt2 = txt + [frmt.format(r[i], *a[i]) for i in range(len(r))]
    result = "\n".join(txt2)
    return result


def _as_pivot(a):
    """present results in pivot table format"""
    if a.dtype.fields is None:
        print("\n...\nStructured array with field names is required")
        return a
    flds = list(a.dtype.names)
    r = np.unique(a[flds[0]])
    c = np.unique(a[flds[1]])
    z = np.zeros((len(r)+1, len(c)+1), dtype=np.float)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in a]
    for i in range(len(a)):
        rr, cc = rc[i]
        z[rr, cc] = a[i][2]
    z[-1, :] = np.sum(z, axis=0)
    z[:, -1] = np.sum(z, axis=1)
    result = _prn(r, c, z, stat_name='Count')
    return result


"""
#out_tbl is a featureclass table name
#txt_file is a text file name
#
#    if not (out_tbl in ['#', '', None]):
#        arcpy.da.NumPyArrayToTable(ctab, out_tbl)
#    if not (txt_file in ['#', '', None]):
#        with open(txt_file, 'w') as f:
#            f.write("Crosstab for ... {}\n".format(in_tbl))
#            f.write(result)
"""


# (1) from featureclass table
def crosstab_tbl(in_tbl, flds=None, as_pivot=True):
    """Derive the unique attributes in a table for all or selected fields.

    Parameters
    ----------
    in_tbl : table
        A featureclass or its table.
    flds : fields
        If None, then all fields in the table are used.
        Make sure that you do not include sequential id fields or all table
        records will be returned.

    Notes
    -----
    None or <null> values in tables are converted to proper nodata values
    depending on the field type.  This is handled by the call to fc_data which
    uses _make_nulls_ to do the work.
    """
    a = npg_io.fc_data(in_tbl)
    if flds is None:
        flds = list(a.dtype.names)
    uni, idx, cnt = np.unique(a[flds], True, False, True)
    out_arr = append_fields(uni, "Counts", cnt, usemask=False)
    if as_pivot:
        return as_pivot(out_arr)
    return out_arr


# (2) from two, 1D numpy ndarrays
def crosstab_rc(row, col, reclassed=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Parameters
    ----------
    row, col : text
        row and column array/field

    Returns
    -------
    ctab : the crosstabulation result as row, col, count array
    rc_ : similar to above, but the row/col unique pairs are combined.
    """
    dt = np.dtype([('row', row.dtype), ('col', col.dtype)])
    rc_zip = list(zip(row, col))
    rc = np.asarray(rc_zip, dtype=dt)
    u, idx, cnts = np.unique(rc, return_index=True, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u['row'], u['col'], cnts)), dtype=rcc_dt)
    # ----
    if reclassed:
        rc2 = np.array(["{}_{}".format(*i) for i in rc_zip])
        u2, idx2, cnts2 = np.unique(rc2, return_index=True, return_counts=True)
        dt = [('r_c', u2.dtype.str), ('cnts', '<i4')]
        rc_ = np.array(list(zip(u2, cnts2)), dtype=dt)
        return rc_
    return ctab


# (3) from a structured array
def crosstab_array(a, flds=None):
    """Frequency and crosstabulation for structured arrays.

    Parameters
    ----------
    a : array
       input structured array

    flds : string or list
       Fields/columns to use in the analysis.  For a single column, a string
       is all that is needed.  Multiple columns require a list of field names.

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
    a = repack_fields(a[flds])  # need to repack fields
    # a = _keep_fields(a, flds)  # alternative to repack_fields
    idx = np.argsort(a, axis=0, order=flds)  # (2) sort
    a_sort = a[idx]
    uniq, counts = np.unique(a_sort, return_counts=True)  # (3) unique, count
    dt = uniq.dtype.descr
    dt.append(('Count', '<i4'))
    fr = np.empty_like(uniq, dtype=dt)
    names = fr.dtype.names
    vals = list(zip(*uniq)) + [counts.tolist()]  # (4) reassemble
    N = len(names)
    for i in range(N):
        fr[names[i]] = vals[i]
    return fr


# ==== Summarize tools ======================================================
# (4) pivot table from 3 numpy ndarrays
def _calc_stats(arr, axis=None, deci=4):
    """Calculate stats for an array of number types, with nodata (nan, None)
    in the column.

    Notes
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
    s = [N, N-cnt, n_sum, n_min, n_max, n_mean, n_med, n_std, n_var]
    s = [np.around(i, deci) for i in s]
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


def col_stats(a, fields=None, deci=2, verbose=False):
    """Calculate statistics for a structured/recarray with or without specified
    fields.  Efforts have been made to check for all possible scenarios, but
    human intelligence should prevail when one decides what to throw at it.

    Parameters
    ----------
    a : array
        A structured/recarray
    fields : list, string or None
      - None,  checks all fields or assumes that the input array is a singleton
      - string, a single field name, if the column names are known
      - list,  a list of field names
    deci : integer
        an attempt to format floats with deci(mal) places

    Requires
    --------
    _numeric_fields_ : function
        returns the numeric fields in a structured/recarray
    _calc_stats : function
        performs the actual field calculations
    """
    s_lst = []
    if isinstance(fields, str):
        fields = [fields]
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
                          'median', 'std', 'var'])
    z = np.zeros((len(col_names),), dtype=dts)
    z['Statistic'] = col_names
    N = len(num_flds)
    for i in range(N):
        fld = num_flds[i]
        z[fld] = s_lst[i]
    if verbose:
        args = ["="*25, "Numeric fields"]
        print("\n{}\nStatistics for... a\n{!s:>32}".format(*args))
        prn_tbl(z)
    return z


def group_stats(a, case_fld=None, num_flds=None, deci=2, verbose=False):
    """Group column statistics.

    Parameters
    ----------
    a : structured/recarray
        Make sure that you know the field names in advance
    case_fld : string, list
        String,  summarized by the unique values in the case_fld.
        List, to further fine-tune the selection or crosstabulation
    num_flds : string, list
        You can limit the input fields accordingly, if you only need a few
        know numeric fields.

    Requires
    --------
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
        u = uniq[i]
        if counts[i] >= 3:
            sub = a[a[case_fld] == u]
            z = col_stats(sub, fields=num_flds, deci=deci)
            if verbose:
                args = ["="*25, u, "Numeric fields"]
                print("\n{}\nStatistics for... a[{}]\n{!s:>32}".format(*args))
                prn_tbl(z)
            results.append([u, z])
        else:
            print("\nToo few cases... ({}) for a[{}]...".format(counts[i], u))
    return results


# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # msg = _demo_()
