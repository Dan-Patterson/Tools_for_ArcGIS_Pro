# -*- coding: utf-8 -*-
"""
pnt_field_calcs
================

Script :   pnt_field_calcs.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-07-25

Purpose :  tools for working ArcGIS Pro field calculator

Useage:
-------


**Requires**
------------


**Notes**
---------

**Basic array information**

"""
# ---- imports, formats, constants -------------------------------------------
import sys
import inspect
from textwrap import dedent, indent
import numpy as np
import arcpy

arcpy.env.overwriteOutput = True

script = sys.argv[0]

# ---- simple functions ------------------------------------------------------

__all__ = []

if len(sys.argv) == 1:
    testing = True
    create_output = False
    in_tbl = "/pnts_2k_normal"
    gdb = "/Table_tools.gdb"
    flder = "/".join(script.split("/")[:-2])
    wrkspace = flder + gdb
    in_tbl = wrkspace + in_tbl
    in_fld = None
else:
    testing = False
    create_output = True
    in_tbl = sys.argv[1]
    in_fld = sys.argv[2]
    exp_key = sys.argv[3]

# ---- Do the work -----------------------------------------------------------
#
desc = arcpy.da.Describe(in_tbl)
wrkspace = desc['path']

# ---- Expression functions
fld_name = in_fld

if exp_key == "cumulative distance":
    from geometry import dist_cumu
    if inspect.isfunction(dist_cumu):
        lines, ln_num = inspect.getsourcelines(dist_cumu)
        code = "".join(["{}".format(line) for line in lines])
        fld_expr = "dist_cumu(!Shape!)"
        fld_name = "Cumu_dist"
    args = [fld_name, fld_expr, code]
elif exp_key == "distance between":
    from geometry import dist_between
    if inspect.isfunction(dist_between):
        lines, ln_num = inspect.getsourcelines(dist_between)
        code = "".join(["{}".format(line) for line in lines])
        fld_expr = "dist_between(!Shape!)"
        fld_name = "Dist_btwn"
    args = [fld_name, fld_expr, code]

fld_name, fld_expr, code = args

arcpy.MakeTableView_management(
        in_table=in_tbl,
        out_view="tbl_view",
        workspace=wrkspace)

if in_fld in (None, "", " "):
    fld_name = fld_name
else:
    fld_name = in_fld
fld_name = arcpy.ValidateFieldName(fld_name)
arcpy.AddField_management(
        "tbl_view",
        field_name=fld_name,
        field_type="DOUBLE",
        field_is_nullable="NULLABLE")

arcpy.CalculateField_management(
        in_table="tbl_view",
        field=fld_name,
        expression=fld_expr,
        code_block=code)

del in_fld, in_tbl, arcpy
# ----------------------------------------------------------------------
# __main__ .... code section

if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    pnts, mesh = _demo()