# -*- coding: UTF-8 -*-
"""
:Script:   field_stats.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-07-08
:Purpose:  Descriptive statistics for tables using numpy.
:
:References:
:
:---------------------------------------------------------------------:
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
