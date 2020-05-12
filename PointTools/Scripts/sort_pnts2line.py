# -*- coding: UTF-8 -*-
"""
script name
===========

Script :   sort_pnts2line.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-06-12

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
`<https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-
continuous-line/37744549#37744549`_.

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
from arcpytools import fc_info, tweet
from arcpytools_pnt import frmt_rec, make_row_format, _col_format, form_
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...),
    :-----------------------------------------------------------------------
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


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
    int_min = np.iinfo(np.int32).min
    float_min = np.finfo(np.float64).min
    str_val = "None"
    nulls = {'Double':float_min, 'Integer':int_min, 'Text':str_val}
    #
    fld_dict = {i.name: i.type for i in arcpy.ListFields(in_tbl)}
    null_dict = {f:nulls[fld_dict[f]] for f in flds}
    a = arcpy.da.TableToNumPyArray(in_table=in_tbl, field_names=flds,
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
#    a = np. array(['1, 2, 3, 4, 5', 'a, b, c', '6, 7, 8, 9',
#                   'd, e, f, g, h', '10, 11, 12, 13'])
    # ---- make random points that should form a line ---- see link
#    from scipy.spatial import kdtree as kd
    x0 = np.linspace(0, 2 * np.pi, 10)
    y0 = np.sin(x0)
    a0 = np.c_[x0, y0]
    d0 = e_dist(a0, a0)
    idx0 = np.arange(len(x0))
    np.fill_diagonal(d0, np.inf)
    # ---- randomize
    idx = np.random.permutation(idx0)
    x = x0[idx]
    y = y0[idx]
    a = np.c_[x, y]  # points array
    d = e_dist(a, a)
    np.fill_diagonal(d, np.inf)
    #
    m = np.argmin(d0, axis=1)
    ft = np.zeros((10,), dtype=[('ID', '<i4'), ('Closest', '<i4')] )
    ft['ID'] = idx0
    ft['Closest'] = m
    return idx, a0, a, d0, d, ft


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
    idx, a0, a, d0, d, ft = _demo()
    frmt = """
    Testing...
    unsorted...
    {}
    sorted.....
    {}
    """
    args = [a0, a]
    tweet(dedent(frmt).format(a0, a))
    tweet(form_(d0, prn=False))
    tweet(form_(d, title="Sorted...", prn=False))
    tweet(frmt_rec(ft, prn=False))
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
