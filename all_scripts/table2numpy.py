# -*- coding: UTF-8 -*-
"""
:Script:   table2numpyarray.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-11
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
    out_folder = sys.argv[3]  # output field name
    out_filename = sys.argv[4]
    out_name = "\\".join([out_folder, out_filename])
    # ---- main tool section
    desc = arcpy.da.Describe(in_tbl)
    args = [in_tbl, in_flds, out_name]
    msg = "Input table.. {}\nfields...\n{}\nOutput arr  {}".format(*args)
    tweet(msg)
    #
    # ---- call section for processing function
    #
    oid = 'OBJECTID'
    in_flds = in_flds.split(";")
    if oid in in_flds:
        vals = in_flds
    else:
        vals = [oid] + in_flds
    #
    # ---- create the field dictionary
    f_info = np.array([[i.name, i.type] for i in arcpy.ListFields(in_tbl)])
    f_dict = {'OBJECTID': -1}
    for f in in_flds:
        if f in f_info[:, 0]:
            n, t = f_info[f_info[:, 0] == f][0]
            if t in ('Integer', 'Short', 'Long'):
                t = np.iinfo(np.int32).min
            elif t in ('Double', 'Float'):
                t = np.nan
            elif t in ('String', 'Text'):
                t = str(None)
            else:
                t = np.iinfo(np.int32).min
            f_dict[n] = t
    # ---- where_clause= skip_nulls=  null_value=)
    arr = arcpy.da.TableToNumPyArray(in_tbl, vals, "#", False, f_dict)
    #
    np.save(out_name, arr)


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    arrs= _demo()
    frmt = "Result...\n{}"
    print(frmt.format(arrs))
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
