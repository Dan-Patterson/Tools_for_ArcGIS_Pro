# -*- coding: UTF-8 -*-
"""
strided_funcs
=============

Script :   strided_funcs.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-07-28

Purpose:  tools for working with numpy arrays using srided functions

Useage :

Generate sequence

def seq(N=100, diff_val=1):
    '''generate a sequence in the range 0, N
    +ve or -ve values in the range of 1 are added
    to the values
    '''
    neg = np.random.random()*-diff_val
    pos = np.random.random()*diff_val
    a = np.arange(0, N, 1) + \
       [[np.random.random(), -np.random.random()][np.random.random <= 0.5]
       for i in range(0, N)]
    return a

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
from arcpytools import fc_info, tweet  #, frmt_rec, _col_format
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
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


def stride(a, win=(3, 3), stepby=(1,1)):
    """Provide a 2D sliding/moving array view.  There is no edge
    correction for outputs.  From arraytools.tools
    """

    err = """Array shape, window and/or step size error.
    Use win=(3,) with stepby=(1,) for 1D array
    or win=(3,3) with stepby=(1,1) for 2D array
    or win=(1,3,3) with stepby=(1,1,1) for 3D
    ----    a.ndim != len(win) != len(stepby) ----
    """
    assert (a.ndim == len(win)) and (len(win) == len(stepby)), err
    shape = np.array(a.shape)  # array shape (r, c) or (d, r, c)
    win_shp = np.array(win)    # window      (3, 3) or (1, 3, 3)
    ss = np.array(stepby)      # step by     (1, 1) or (1, 1, 1)
    newshape = tuple(((shape - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides).squeeze()
    return a_s


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
             'Integer':np.iinfo(np.int32).min,
             'OID':np.iinfo(np.int32).min,
             'String':None}
    #
    fld_dict = {i.name: i.type for i in arcpy.ListFields(in_tbl)}
    null_dict = {f:nulls[fld_dict[f]] for f in flds}
    a = arcpy.da.TableToNumPyArray(in_table=in_tbl,
                                   field_names=flds,
                                   skip_nulls=False,
                                   null_value=null_dict)
    return a


def strided_func(a, step=3, func='mean'):
    """ strides a numeric array in preparation for numeric calculations.

    `a` : array of floats form most functions
        numeric array with shape (N,)
    `step` : integer
        the step window size such as a 3, 5 or 7 moving window
    """
    def mode(b, axis=None):
        """Calculate the modal value and optional count
        """
        modes, cnts = np.unique(b, return_counts=True, axis=axis)
        idx = np.argmax(cnts)
        return modes[idx]  #  counts[index]

    start = step // 2  # integer division to determine the start for filling
    b = stride(a, win=(step,), stepby=(1,))
    out_array = np.zeros((a.shape[0],), dtype=a.dtype)
    out_array.fill(np.nan)
    if func == 'mean':
        out_array[start: -start] = np.nanmean(b, axis=1)
    elif func == 'median':
        out_array[start: -start] = np.nanmedian(b, axis=1)
    elif func == 'min':
        out_array[start: -start] = np.nanmin(b, axis=1)
    elif func == 'max':
        out_array[start: -start] = np.nanmax(b, axis=1)
    elif func == 'sum':
        out_array[start: -start] = np.nansum(b, axis=1)
    elif func == 'mode':
        out_array[start: -start] = [mode(i) for i in b]
    elif func == 'trend':  # returns -1, 0, 1 for down, flat, up
        out_array[start: -start] = [mode(np.sign(np.diff(i))) for i in b]
    return out_array


def form_output(in_tbl, in_arr, out_fld="Result_",
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
    del_fld = True
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
    """
    tbl = "Table_tools.gdb/pnts_2k_normal"
    in_tbl = "/".join(script.split("/")[:-2] + [tbl])
    #
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    in_fld = 'Sequences'  # 'Norm'  # 'Ys'
    out_fld = 'Result_fld'
    in_flds = [oid_fld, in_fld]   # OBJECTID, plus another field
    in_arr = tbl_2_nparray(in_tbl, in_flds)
    func = 'mean'  #np.random.choice(c)
    win_size = 5
    xtend = False
    return in_tbl, in_arr, in_fld, out_fld, func, win_size, xtend


def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_fld = sys.argv[2]
    out_fld = sys.argv[3]  # output field name
    func = sys.argv[4]
    win_size = int(sys.argv[5])

    # ---- main tool section
    desc = arcpy.da.Describe(in_tbl)
    # ---- main tool section
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    flds = [oid_fld, in_fld]
    tbl_path = desc['path']
    fnames = [i.name for i in arcpy.ListFields(in_tbl)]
    if out_fld in fnames:
        out_fld += 'dup'
    out_fld = arcpy.ValidateFieldName(out_fld, tbl_path)
    args = [in_tbl, in_fld, out_fld, tbl_path]
    msg = "in_tbl {}\nin_fld {}\nout_fld  {}\ntbl_path  {}".format(*args)
    tweet(msg)
    #
    # ---- call section for processing function
    #
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    # ---- remove the selection by calling the table
    in_tbl  = desc['catalogPath']
    #
    flds = [oid_fld, in_fld]
    in_arr = tbl_2_nparray(in_tbl, flds)
    tweet("{!r:}".format(in_arr))
    xtend = True
    return in_tbl, in_arr, in_fld, out_fld, func, win_size, xtend

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    in_tbl, in_arr, in_fld, out_fld, func, win_size, xtend = _demo()
else:
    testing = False
    in_tbl, in_arr, in_fld, out_fld, func, win_size, xtend = _tool()
#
if not testing:
    tweet('Some message here...')

# ---- reassemble the table for extending ----

a = in_arr[in_fld]
result = strided_func(a, step=win_size, func=func)

out_array = form_output(in_tbl,
                        in_arr,
                        out_fld=out_fld,
                        vals=result,
                        idx=0,
                        xtend=xtend)
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    a = _demo()
