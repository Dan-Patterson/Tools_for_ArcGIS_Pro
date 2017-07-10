# -*- coding: UTF-8 -*-
"""
:Script:   field_stats.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-07-08
:Purpose:  Descriptive statistics for tables using numpy.
:
:References:
:  https://github.com/numpy/numpy/blob/master/numpy/lib/nanfunctions.py
:  def _replace_nan(a, val):
:      mask = np.isnan(a)
:      if mask is not None:
:          np.copyto(a, val, where=mask)
:      return a, mask
:
:  a = [1, 2, np.nan, 3, np.nan, 4]
:  arr, mask = _replace_nan(a, 0)  # for mean
:  # array([False, False,  True, False,  True, False], dtype=bool)
:  a  # array([ 1.,  2.,  0.,  3.,  0.,  4.])  # returned

:  def _divide_by_count(a, b, out=None):
:      '''if it is an array '''
:      return np.divide(a, b, out=a, casting='unsafe')

nanmean uses this ---
arr, mask = _replace_nan(a, 0)
if mask is None:
    return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
# otherwise
cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims)
tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
avg = _divide_by_count(tot, cnt, out=out)
return avg

def nanvar():
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.var(arr)
    # compute the mean
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=_keepdims)
    avg = np.sum(arr, axis=axis, dtype=dtype, keepdims=_keepdims)
    avg = _divide_by_count(avg, cnt)
    # Compute squared deviation from mean.
    np.subtract(arr, avg, out=arr, casting='unsafe')
    arr = _copyto(arr, 0, mask)
    sqr = np.multiply(arr, arr, out=arr)
    # compute the variance
    var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    dof = cnt - ddof
    var = _divide_by_count(var, dof)
    return var
: ---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

def _replace_nan(a, val):
    mask = np.isnan(a)
    if mask is not None:
        np.copyto(a, val, where=mask)
    return a, mask

def skewness(arr):
    '''Momental and unbiased skewness'''
    skew_m = None
    skew_u = None
    n = float(len(arr))
    if n < 3:
        return a
    a, mask = _replace_nan(arr, 0)  # at least 2 samples found
    x_m = np.nanmean(a)
    diff = a - x_m
    x_2 = np.sum(np.power(diff, 2))
    x_3 = np.sum(np.power(diff, 3))
    var_x = x_2 / n
    std_x = math.sqrt(var_x)
    vars_x = x_2 / (n-1)
    stds_x = math.sqrt(vars_x)
    skew_m = x_3 / (n * (std_x**3))
    if n > 2:
        skew_u = (x_3 / (n * (stds_x**3)))*((n**2)/((n-1)*(n-2)))
    return [skew_m, skew_u]



# -----------------------------------------------------------------------------
# functions

def tweet(msg):
    """Produce a message (msg)for both arcpy and python
    """
    m = "{}".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


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
    args = (col_names, np.nansum(b, axis=0),
            np.nanmin(b, axis=0), np.nanmax(b, axis=0),
            np.nanmedian(b, axis=0), np.nanmean(b, axis=0),
            np.nanstd(b, axis=0), np.nanvar(b, axis=0))
    return col_names, args


def stats_tbl(col_names, args):
    """Produce the output table
    :   ('N_', '<i4'), ('N_nan', '<i4')
    """
    d = [(i, '<f8') for i in ['Sum', 'Min', 'Max', 'Med', 'Avg', 'Std', 'Var']]
    dts = [('Fld', '<U15')] + d
    rows = len(col_names)
    cols = len(dts)
    z = np.empty(shape=(rows,), dtype=dts)
    for i in range(cols):
        z[z.dtype.names[i]] = args[i]
    return z


if len(sys.argv) == 1:
    in_fc = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_1000_nan'
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

tweet("\n{}\nSaving results to .... {}".format("-"*60, out_tbl))
tweet("Stats results...\n{}\n{}".format(z.dtype.names, z))

if out_tbl is not None:
    arcpy.da.NumPyArrayToTable(z, out_tbl)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
