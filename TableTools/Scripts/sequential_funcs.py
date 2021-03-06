# -*- coding: UTF-8 -*-
"""
sequential_funcs
================

Script:   sequential_funcs.py

Author:   Dan.Patterson@carleton.ca

Modified: 2018-12-28

Purpose :
    Calculating sequential values for fields in geodatabase tables

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy
from arcpytools import fc_info, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def has_nulls(a):
    """Check to see if nulls are in the array passed from the featureclass
    """
    #
    a_kind = a.dtype.kind
    if a_kind == 'i':
        m = a == np.iinfo(np.int32).min
    elif a_kind == 'f':
        m = np.isnan(a)
    else:
        m = a == None
    return m


def tbl_2_nparray(in_tbl, flds):
    """Form the TableToNumPyArray to account for nulls for various dtypes.
    This is essentially a shortcut to `arcpy.da.TableToNumPyArray`

    Requires
    --------
    `in_tbl` :
        table, or featureclass table name
    `flds` :
        list of field names
    `skip_nulls` = False :
        set within function
    `null_value` :
        determined from the dtype of the array...
        otherwise you may as well do it manually

    Source
    ------
    arraytools, apt.py module
    """
    nulls = {'Double':np.nan,
             'Single':np.nan,
             'Integer':np.iinfo(np.int32).min,
             'OID':np.iinfo(np.int32).min,
             'String':"None"}
    #
    fld_dict = {i.name: i.type for i in arcpy.ListFields(in_tbl)}
    null_dict = {f:nulls[fld_dict[f]] for f in flds}
    a = arcpy.da.TableToNumPyArray(in_table=in_tbl,
                                   field_names=flds,
                                   skip_nulls=False,
                                   null_value=null_dict)
    return a


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


def percent(a):
    """value percentage"""
    m = has_nulls(a)
    if not np.alltrue(m):
        a = np.ma.MaskedArray(a, mask=m)
        return (a/(np.ma.sum(a) * 1.0)) * 100.
    else:
        return (a/(np.sum(a) * 1.)) * 100.


def seq_diff(a):
    """Sequential diffs"""
    return a[1:] - a[:-1]


def seq_number(a):
    """Sequentially number the class values in a field
    """
    uni = np.unique(a)
    max_sze = [len(i) for i in uni]
    out = np.chararray(len(a), max_sze + 5, True)
    for u in uni:
        idx = np.where(a == u)[0]
        cnt = 0
        for i in idx:
            out[i] = "{}{:02.0f}".format(u, cnt)
            cnt += 1
    return out


def val_diff(a, val):
    """diff from a value"""
    return a - val


def z_score(a):
    """Z-scores"""
    return mean_diff(a)/np.nanstd(a)


def seq_count(a):
    """See `running_count` in arraytools.tools for a fuller version
    """
    idx = a.argsort(kind='mergesort')
    s_a = a[idx]
    neq = np.where(s_a[1:] != s_a[:-1])[0] + 1
    run = np.ones(a.shape, int)
    run[neq[0]] -= neq[0]
    run[neq[1:]] -= np.diff(neq)
    out = np.empty_like(run)
    out[idx] = run.cumsum()
    z = np.array(["{}_{:0>3}".format(*i) for i in list(zip(a, out))])
    return z


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
    in_fld = 'Unif'  # 'Sequences2'  #'Ys'
    del_fld = True
    out_fld = 'Result_fld'
    in_flds = [oid_fld, in_fld]   # OBJECTID, plus another field
    in_arr = tbl_2_nparray(in_tbl, in_flds)
    c = np.array(['cumulative sum', 'diff from max',
                  'diff from mean', 'diff from median',
                  'diff from min', 'diff from value',
                  'percent', 'sequential diff',
                  'sequential number',
                  'z_score'])
    func = 'percent'  #np.random.choice(c)
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
    in_arr = tbl_2_nparray(in_tbl, flds)
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
elif func == 'percent':
    result = percent(a)
    idx = 0
elif func == 'sequential diff':
    result = seq_diff(a)  # sequential diff call
    idx = 1
elif func == 'sequential number':
    result = seq_number(a)
    idx = 0
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
#    print("Script... {}".format(script))
#    tbl = r"C:\Git_Dan\arraytools\array_tools_testing\Data\pnts_2000.npy"
