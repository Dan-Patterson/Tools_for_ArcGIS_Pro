# -*- coding: UTF-8 -*-
"""
:Script:   field_statistics.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-11
:Purpose:  Descriptive statistics for tables using numpy.
:
:References:
:  https://github.com/numpy/numpy/blob/master/numpy/lib/nanfunctions.py
:  _replace_nan(a, val) -  mask = np.isnan(a) - to get the mask
:
:  a = [1, 2, np.nan, 3, np.nan, 4]
:  _, mask = _replace_nan(a, 0)  # for mean
:  mask = array([False, False,  True, False,  True, False], dtype=bool)
:
: ---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -
script = sys.argv[0]  # print this should you need to locate the script


# ---- skewness and kurtosis section -----------------------------------------

def skew_kurt(a, avg, var_x, std_x, col=True, mom='both'):
    """Momental and unbiased skewness
    :Emulates the nan functions approach to calculating these parameters
    :when data contains nan values.
    :Requires:
    :---------
    :  a - an array of float/double values where there are at least 3 non-nan
    :      numbers in each column.  This is not checked since this situation
    :      should never arise in real world data sets that have been checked.
    :  moment - both, skew or kurt  to return the moments
    :Notes:
    :------
    : a= np.arange(16.).reshape(4,4)
    : mask = [0, 5, 10, 15]
    : masked_array = np.where(a == mask, np.nan, a)
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


# -----------------------------------------------------------------------------
# functions

def tweet(msg):
    """Produce a message (msg)for both arcpy and python
    """
    m = "{}".format(msg)
    arcpy.AddMessage(m)
    print(m)


# (1) frmt_rec .... code section ... from frmts.py in arraytools
#  frmt_rec requires _col_format
def _col_format(a, c_name="c00", deci=0):
    """Determine column format given a desired number of decimal places.
    :  Used by frmt_struct.
    :  a - a column in an array
    :  c_name - column name
    :  deci - desired number of decimal points if the data are numeric
    :Notes:
    :-----
    :  The field is examined to determine whether it is a simple integer, a
    :  float type or a list, array or string.  The maximum width is determined
    :  based on this type.
    :  Checks were also added for (N,) shaped structured arrays being
    :  reformatted to (N, 1) shape which sometimes occurs to facilitate array
    :  viewing.  A kludge at best, but it works for now.
    """
    a_kind = a.dtype.kind
    if a_kind in ('i', 'u'):  # ---- integer type
        w_, m_ = [':> {}.0f', '{:> 0.0f}']
        col_wdth = len(m_.format(a.max())) + 1
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = w_.format(col_wdth, 0)
    elif a_kind == 'f' and np.isscalar(a[0]):  # ---- float type with rounding
        w_, m_ = [':> {}.{}f', '{:> 0.{}f}']
        a_max, a_min = np.round(np.sort(a[[0, -1]]), deci)
        col_wdth = max(len(m_.format(a_max, deci)),
                       len(m_.format(a_min, deci))) + 1
        col_wdth = max(len(c_name), col_wdth) + 1
        c_fmt = w_.format(col_wdth, deci)
    else:  # ---- lists, arrays, strings. Check for (N,) vs (N,1)
        if a.ndim == 1:  # ---- check for (N, 1) format of structured array
            a = a[0]
        col_wdth = max([len(str(i)) for i in a])
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = "!s:>" + "{}".format(col_wdth)
    return c_fmt, col_wdth


def frmt_rec(a, deci=2, use_names=True, prn=True):
    """Format a structured array with a mixed dtype.
    :Requires
    :-------
    : a - a structured/recarray
    : deci - to facilitate printing, this value is the number of decimal
    :        points to use for all floating point fields.
    : _col_format - does the actual work of obtaining a representation of
    :  the column format.
    :Notes
    :-----
    :  It is not really possible to deconstruct the exact number of decimals
    :  to use for float values, so a decision had to be made to simplify.
    """
    dt_names = a.dtype.names
    N = len(dt_names)
    c_names = [["C{:02.0f}".format(i) for i in range(N)], dt_names][use_names]
    # ---- get the column formats from ... _col_format ----
    dts = []
    wdths = []
    pair = list(zip(dt_names, c_names))
    for i in range(len(pair)):
        fld, nme = pair[i]
        c_fmt, col_wdth = _col_format(a[fld], c_name=nme, deci=deci)
        dts.append(c_fmt)
        wdths.append(col_wdth)
    row_frmt = " ".join([('{' + i + '}') for i in dts])
    hdr = ["!s:>" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = " ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = "--n--" + hdr2.format(*c_names)
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    # ---- check for structured arrays reshaped to (N, 1) instead of (N,) ----
    len_shp = len(a.shape)
    idx = 0
    for i in range(a.shape[0]):
        if len_shp == 1:  # ---- conventional (N,) shaped array
            row = " {:03.0f} ".format(idx) + row_frmt.format(*a[i])
        else:             # ---- reformatted to (N, 1)
            row = " {:03.0f} ".format(idx) + row_frmt.format(*a[i][0])
        idx += 1
        txt.append(row)
    msg = "\n".join([i for i in txt])
    if prn:
        print(msg)
    else:
        return msg


def cal_stats(in_fc, col_names):
    """Calculate stats for an array of double types, with nodata (nan, None)
    :  in the column.
    :Requires:
    :---------
    : in_fc - input featureclass or table
    : col_names - the columns... numeric (floating point, double)
    :
    :Notes:
    :------  see the args tuple for examples of nan functions
    :  np.nansum(b, axis=0)   # by column
    :  np.nansum(b, axis=1)   # by row
    :  c_nan = np.count_nonzero(~np.isnan(b), axis=0) count nan if needed
    """
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, col_names)  # "*")
    b = a.view(np.float).reshape(len(a), -1)
    if len(a.shape) == 1:
        ax = 0
    else:
        ax = [1, 0][True]  # ax = [1, 0][colwise]  colwise= True
    mask = np.isnan(b)
    cnt = np.sum(~mask, axis=ax, dtype=np.intp, keepdims=False)
    n_sum = np.nansum(b, axis=0)
    n_mean = np.nanmean(b, axis=0)
    n_var = np.nanvar(b, axis=0)
    n_std = np.nanstd(b, axis=0)
    sk, kurt = skew_kurt(b, avg=n_mean, var_x=n_var, std_x=n_std,
                         col=True, mom='both')
    args = (col_names, cnt, n_sum, np.nanmin(b, axis=0), np.nanmax(b, axis=0),
            np.nanmedian(b, axis=0), n_mean, n_std, n_var, sk, kurt)
    return col_names, args


def stats_tbl(col_names, args):
    """Produce the output table
    :   ('N_', '<i4'), ('N_nan', '<i4')
    """
    d = [(i, '<f8')
         for i in ['Sum', 'Min', 'Max', 'Med', 'Avg',
                   'Std', 'Var', 'Skew', 'Kurt']]
    dts = [('Field', '<U15'), ('N', '<i4')] + d
    rows = len(col_names)
    cols = len(dts)
    z = np.empty(shape=(rows,), dtype=dts)
    for i in range(cols):
        z[z.dtype.names[i]] = args[i]
    return z


if len(sys.argv) == 1:
    in_fc = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_2K_normal'
    flds = arcpy.ListFields(in_fc)
    col_names = [fld.name for fld in flds if fld.type == 'Double']
    out_tbl = None
else:
    in_fc = sys.argv[1]
    col_names = sys.argv[2]
    col_names = col_names.split(';')
    out_tbl = sys.argv[3]

col_names, args = cal_stats(in_fc, col_names)  # calculate statistics
z = stats_tbl(col_names, args)                 # produce the table

msg = frmt_rec(z, prn=False)  # fancy printout
tweet("\n{}\nSaving results to .... {}".format("-"*60, out_tbl))
tweet("Stats results...\n{}".format(msg))

if not (out_tbl in (None, '#', '', 'None')):
    arcpy.da.NumPyArrayToTable(z, out_tbl)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
