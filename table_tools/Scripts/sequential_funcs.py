# -*- coding: UTF-8 -*-
"""
sequential_funcs
================

Script:   sequential_funcs.py
Author:   Dan.Patterson@carleton.ca
Modified: 2018-05-19
Purpose :
    Calculating sequential values for fields in geodatabase tables
Useage :

:References:
:  http://pro.arcgis.com/en/pro-app/arcpy/functions/
:       numpyarraytoraster-function.htm
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy
from arcpytools import fc_info, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def cum_sum(a):
    """Cumulative sum"""
    return np.nancumsum(a)

def max_diff(a):
    """diff from max"""
    return a - np.nanmax(a)


def mean_diff(a):
    """diff from mean"""
    return a - np.nanmean(a)


def median_diff(a):
    """diff from median"""
    return a - np.nanmedian(a)


def min_diff(a):
    """diff from min"""
    return a - np.nanmin(a)


def seq_diff(a):
    """Sequential diffs"""
    return a[1:] - a[:-1]


def val_diff(a, val):
    """diff from a value"""
    return a - val


def z_score(a):
    """Z-scores"""
    return mean_diff(a)/np.nanstd(a)


def form_output(in_tbl, in_arr, out_fld="Result_", del_fld=True,
                vals=None, idx=0, xtend=False):
    """Form the output table given a field name and join field

    Requires:
    ---------

    tbl :
        input table
    fld_name :
        output field names, should contain OBJECTID and desired output field
    vals :
        values for output field
    sze :
        string representation of output field
    idx :
        index to start values from... usually 0 or 1 (ie for sequential)

    """
    desc = arcpy.da.Describe(in_tbl)
    tbl_path = desc['path']
    oid_fld = desc['OIDFieldName']   # 'OBJECTID'
    fnames = [i.name for i in arcpy.ListFields(in_tbl)]
    if del_fld in ('True', 'true', True, 1):
        del_fld = True
    else:
        del_fld = False
    if out_fld not in fnames:
        out_fld = out_fld
    elif out_fld in fnames and del_fld:
        arcpy.DeleteField_management(in_tbl, out_fld)
        tweet("\nDeleting field {}".format(out_fld))
    else:
        out_fld += 'dup'
    out_fld = arcpy.ValidateFieldName(out_fld, tbl_path)
    #
    sze = vals.dtype.str
    dt = [('IDs', '<i4'), (out_fld, sze)]  # ie '<f8'
    out_array = np.zeros((in_arr.shape[0],), dtype=dt)
    out_array['IDs'] = in_arr[oid_fld]
    out_array[out_fld][idx:] = vals
    if xtend:
        arcpy.da.ExtendTable(in_tbl, oid_fld, out_array, 'IDs')
    return out_array


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    Requires:
        arcpytools fc_info, tweet
    """
    tbl = "Table_tools.gdb/pnts_2k_normal"
    in_tbl = "/".join(script.split("/")[:-2] + [tbl])
    #
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    in_fld = 'Ys'
    del_fld = True
    out_fld = 'Result_fld'
    in_flds = [oid_fld, in_fld]   # OBJECTID, plus another field
    in_arr = arcpy.da.TableToNumPyArray(in_tbl, in_flds)
    c = np.array(['cumulative sum', 'diff from max',
                  'diff from mean', 'diff from median',
                  'diff from min', 'diff from value', 'sequential diff',
                  'z_score'])
    func = np.random.choice(c)
    xtend = False
    val = None
    return in_tbl, in_arr, in_fld, out_fld, del_fld, func, xtend, val

def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_fld = sys.argv[2]
    func = sys.argv[3]
    out_fld = sys.argv[4]  # output field name
    del_fld = sys.argv[5]
    val = sys.argv[6]
    #
    # ---- main tool section
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    flds = [oid_fld, in_fld]
    in_arr = arcpy.da.TableToNumPyArray(in_tbl, flds)
    tweet("{!r:}".format(in_arr))
    xtend = True
    return in_tbl, in_arr, in_fld, out_fld, del_fld, func, xtend, val


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
#
if len(sys.argv) == 1:
    testing = True
    in_tbl, in_arr, in_fld, out_fld, del_fld, func, xtend, val = _demo()
else:
    testing = False
    in_tbl, in_arr, in_fld, out_fld, del_fld, func, xtend, val = _tool()

a = in_arr[in_fld]  # do stuff with array

if func == 'cumulative sum':
    result = cum_sum(a)  # sequential diff call
    idx = 0
elif func == 'diff from max':
    result = max_diff(a)
    idx = 0
elif func == 'diff from mean':
    result = mean_diff(a)
    idx = 0
elif func == 'diff from median':
    result = median_diff(a)
    idx = 0
elif func == 'diff from min':
    result = min_diff(a)
    idx = 0
elif func == 'sequential diff':
    result = seq_diff(a)  # sequential diff call
    idx = 1
elif func == 'diff from value':
    idx = 0
    val_orig = val
    try:    val = int(val)
    except:    val = 0
    try:    val = float(val)
    except:    val = 0
    finally:
        frmt = "Difference value entered... {!r:}... Value used... {!r:}"
        tweet(frmt.format(val_orig, val))
        pass
    result = val_diff(a, val)
elif func == 'z_score':
    result = z_score(a)
    idx = 0
else:
    result = seq_diff(a)
    idx = 1
#
# ---- reassemble the table for extending ----
out_array = form_output(in_tbl,
                        in_arr,
                        out_fld=out_fld,
                        del_fld=del_fld,
                        vals=result,
                        idx=idx,
                        xtend=xtend)
msg = """
Processing... {}
function..... {}
input field.. {}
output field. {}
"""

tweet(msg.format(in_tbl, func, in_fld, out_fld))
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    print("Script... {}".format(script))
