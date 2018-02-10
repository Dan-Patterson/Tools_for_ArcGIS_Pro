# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
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
    print(arcpy.GetMessages())


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
    in_flds = sys.argv[2]
    out_fld = sys.argv[3]

    if ';' in in_flds:
        in_flds = in_flds.split(';')
    else:
        in_flds = [in_flds]

    desc = arcpy.da.Describe(in_tbl)
    tbl_path = desc['path']
    fnames = [i.name for i in arcpy.ListFields(in_tbl)]
    if out_fld in fnames:
        out_fld += 'dup'
    out_fld = arcpy.ValidateFieldName(out_fld, tbl_path)
    args = [in_tbl, in_flds, out_fld, tbl_path]
    msg = "in_tbl {}\nin_fld {}\nout_fld  {}\ntbl_path  {}".format(*args)
    tweet(msg)
    oid = 'OBJECTID'
    vals = [oid] + in_flds
    arr = arcpy.da.TableToNumPyArray(in_tbl, vals)
    tweet("{!r:}".format(arr))


arcpy.da.ExtendTable(in_tbl, 'OBJECTID', out_array, 'OBJECTID')


def _demo():
    """
    : -
    """
    pass
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    _demo()

