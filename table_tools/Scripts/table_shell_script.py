# -*- coding: UTF-8 -*-
"""
script name
===========

Script :   ......py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-06-04

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
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
             'String':"None"}
    #
    fld_dict = {i.name: i.type for i in arcpy.ListFields(in_tbl)}
    null_dict = {f:nulls[fld_dict[f]] for f in flds}
    a = arcpy.da.TableToNumPyArray(in_table=in_tbl,
                                   field_names=flds,
                                   skip_nulls=False,
                                   null_value=null_dict)
    return a


def your_func_here():
    """ this is the place"""
    pass


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    """
    a = np. array(['1, 2, 3, 4, 5', 'a, b, c', '6, 7, 8, 9',
                   'd, e, f, g, h', '10, 11, 12, 13'])
    return a


def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_fld = sys.argv[2]
    out_fld = sys.argv[3]  # output field name

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
    #in_arr = arcpy.da.TableToNumPyArray(in_tbl, vals)  # old
    in_arr = tbl_2_nparray(in_tbl, flds)  # produce the table
    #
    tweet("{!r:}".format(in_arr))
    #
    a0 = in_arr[in_fld]
    #
    # do stuff here ********************************
    #
    sze = a0.dtype.str
    # ---- reassemble the table for extending
    dt = [('IDs', '<i8'), (out_fld, sze)]
    out_array = np.copy(in_arr.shape[0])
    out_array[out_fld] = a0  # result goes here
    out_array.dtype = dt
    arcpy.da.ExtendTable(in_tbl, 'OBJECTID', out_array, 'IDs')

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    a = _demo()
    frmt = "Testing...\n{}"
    print(frmt.format(a))
else:
    testing = False
    _tool()
#
if not testing:
    tweet('Some message here...')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    a = _demo()
