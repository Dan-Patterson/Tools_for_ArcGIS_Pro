# -*- coding: UTF-8 -*-
"""
:Script:   field_stats.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-07-08
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

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

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
    sk, kurt = skew_kurt(b, avg=n_mean, var_x=n_var, std_x=n_std, col=True, mom='both')
    args = (col_names, n_sum, np.nanmin(b, axis=0), np.nanmax(b, axis=0),
            np.nanmedian(b, axis=0), n_mean, n_std, n_var, sk, kurt)
    return col_names, args


def stats_tbl(col_names, args):
    """Produce the output table
    :   ('N_', '<i4'), ('N_nan', '<i4')
    """
    d = [(i, '<f8')
         for i in ['Sum', 'Min', 'Max', 'Med', 'Avg',
                   'Std', 'Var', 'Skew', 'Kurt']]
    dts = [('Fld', '<U15')] + d
    rows = len(col_names)
    cols = len(dts)
    z = np.empty(shape=(rows,), dtype=dts)
    for i in range(cols):
        z[z.dtype.names[i]] = args[i]
    return z


if len(sys.argv) == 1:
    in_fc = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_2000_norm'
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
