# -*- coding: UTF-8 -*-
"""
:Script:   numpyarray2table.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-11
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
: - Derived from python snippet output...
:  in_arr = 'C:/Temp/x.npy'
:  out_gdb = 'C:/GIS/Tools_scripts/Table_tools/Table_tools.gdb'
:  out_name = 'sample_1000_npy'
:  arcpy.Tabletools.NumPyArrayToTable(in_arr, out_gdb, out_name)
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
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
    in_arr = sys.argv[1]
    out_gdb = sys.argv[2]
    out_name = sys.argv[3]
    #
    arcpy.env.workspace = out_gdb
    tbls = arcpy.ListTables()
    out_name = arcpy.ValidateTableName(out_name)
    if out_name in tbls:
        out_name += '_dup'
    #
    # ---- call section for processing function
    #
    a = np.load(in_arr)
    in_table = "\\".join([out_gdb, out_name])
    # ---- where_clause= skip_nulls=  null_value=)
    arcpy.da.NumPyArrayToTable(a, in_table)
    arcpy.MakeTableView_management(in_table, out_name)
    #
    args = [in_arr, out_gdb, out_name]
    msg = """
    :------------------------------------------------------------
    Input array... {}
    Output gdb.... {}
    Output name... {}

    Conversion complete...
    Add the table manually if you want to see it...
    :------------------------------------------------------------
    """
    msg = dedent(msg).format(*args)
    tweet(msg)


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    arrs = _demo()
    frmt = "Result...\n{}"
    print(frmt.format(arrs))
else:
    testing = False
    _tool()
#
if not testing:
    tweet('\nConversion done...')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    a = _demo()
