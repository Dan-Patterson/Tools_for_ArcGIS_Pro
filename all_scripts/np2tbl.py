# -*- coding: UTF-8 -*-
"""
np2tbl
======

Script  : np2tbl.py

Author  :   Dan.Patterson@carleton.ca

Modified: 2018-09-23

Purpose:  tools for working with numpy arrays

References:
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.

Derived from python snippet output...

>>> in_arr = 'C:/Temp/x.npy'
>>> out_gdb = 'C:/GIS/Tools_scripts/Table_tools/Table_tools.gdb'
>>> out_name = 'sample_1000_npy'
>>> arcpy.Tabletools.NumPyArrayToTable(in_arr, out_gdb, out_name)

---------------------------------------------------------------------
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
from arcpy import (AddMessage, ListTables, ValidateTableName,
                   MakeTableView_management)
from arcpy.da import NumPyArrayToTable
from arcpy.geoprocessing import env

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    AddMessage(m)
    print(m)


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    pth = script.split("/")[:-2]
    pth = "/".join(pth) + "/Data/sample_20.npy"
    a = np.load(pth)
    frmt = "Result...\n{}"
    print(frmt.format(a))
else:
    testing = False
    in_arr = sys.argv[1]
    out_name = sys.argv[2]
    out_gdb = sys.argv[3]
    make_tbl_view = sys.argv[4]
    env.workspace = out_gdb
    tbls = ListTables()
    out_name = ValidateTableName(out_name)
    if tbls is not None:
        if out_name in tbls:
            out_name += '_dup'
    out_tbl = out_gdb + "/" + out_name
    # ---- call section for processing function
    #
    a = np.load(in_arr)
    NumPyArrayToTable(a, out_tbl)  # create the table
    if make_tbl_view in (True, 'True', 1):
        MakeTableView_management(out_tbl, out_name)
    args = [in_arr, out_gdb, out_name]
    msg = """
    :------------------------------------------------------------

    Input array... {}
    Output gdb.... {}
    Output name... {}

    Conversion complete...
    Add the table manually if you want to see it...

    You need to refresh the geodatabase first since there is no
    autorefresh

    :------------------------------------------------------------
    """
    msg = dedent(msg).format(*args)
    tweet(msg)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
