# -*- coding: utf-8 -*-
"""
field_calculator_tools
======================

Script :   field_calculator_tools.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-07-25

Purpose :  tools for working ArcGIS Pro field calculator

Useage:
-------



**Requires**
------------
  see import section and __init__.py in the `arraytools` folder

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
    in_tbl = "/Carp_AOI"
    gdb = "/Field_calculator_tools.gdb"
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
if exp_key == "area sq km":
     args = ["Area_sqkm", "!Shape!.getArea('GEODESIC','SQUAREKILOMETERS')",
             "#"]
elif exp_key == "leng km":
    args = ["Leng_km", "!Shape!.getLength('GEODESIC','KILOMETERS')", "#"]
elif exp_key in ("sum angles", "min angle", "max angle"):
    from angles_ import angles_poly
    if inspect.isfunction(angles_poly):
        lines, ln_num = inspect.getsourcelines(angles_poly)
        code = "".join(["{}".format(line) for line in lines])
        if exp_key == "sum angles":
            fld_expr = "angles_poly(!Shape!, kind='sum')"
            fld_name = "Angle_sum"
        elif exp_key == "min angle":
            fld_expr = "angles_poly(!Shape!, kind='min')"
            fld_name = "Angle_min"
        elif exp_key == "max angle":
            fld_expr = "angles_poly(!Shape!, kind='max')"
            fld_name = "Angle_max"
    args = [fld_name, fld_expr, code]
elif exp_key in ("cumu_dist"):
    import cumu_dist
    from cumu_dist import dist_cumu
    if inspect.isfunction(dist_cumu):
        lines, ln_num = inspect.getsourcelines(dist_cumu)
        code = "".join(["{}".format(line) for line in lines])
        if exp_key == "cumu_dist":
            fld_expr = "dist_cumu(!Shape!, is_first=True)"
            fld_name = "Cumu_dist"
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