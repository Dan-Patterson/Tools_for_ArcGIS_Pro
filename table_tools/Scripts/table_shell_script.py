# -*- coding: UTF-8 -*-
"""
:Script:   array2raster.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-12
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:  http://pro.arcgis.com/en/pro-app/arcpy/functions/
:       numpyarraytoraster-function.htm
:---------------------------------------------------------------------:
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


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


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
    oid = 'OBJECTID'
    vals = [oid] + in_fld
    in_arr = arcpy.da.TableToNumPyArray(in_tbl, vals)
    tweet("{!r:}".format(arr))
    #
    a0 = in_arr[in_fld]
    # do stuff here
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
    frmt = "Result...\n{}"
    print(frmt.format(a))
else:
    testing = False
    _tool()
#
if not testing:
    print('Concatenation done...')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    a = _demo()
