# -*- coding: UTF-8 -*-
"""
=========
frequency
=========

Script : frequency.py

Author :  Dan_Patterson@carleton.ca

Modified : 2019-02-23

Purpose :
    To supplant the Frequency tool for those that don't have an
    advanced license.

Useage :  Load the toolbox into Pro and run the tool script from there.

References
----------
`<http://desktop.arcgis.com/en/arcmap/latest/tools/analysis-toolbox/
frequency.htm>`_.

Notes
-----

>>> to_array = arcpy.da.TableToNumPyArray(r"C:\folder\sample.dbf", "*")
>>> arcpy.da.NumPyArrayToTable(from_array, r"C:\folder_tbl\test.gdb\out")

Dev Info

  tbx - C:\GIS\Tools_scripts\Table_tools\Table_tools.tbx
  script - C:\GIS\Tools_scripts\Table_tools\Scripts\frequency.py
  arcpy.Tabletools.Frequency("polygon_demo",
             r"C:\GIS\Tools_scripts\Table_tools\Table_tools.gdb\f2",
             "Test;main_part", None)
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy

import numpy.lib.recfunctions as rfn

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def freq(a, flds=None):
    """Frequency and crosstabulation

    Parameters
    ----------
    a : array
        A structured array
    flds : field
        Fields to use in the analysis

    Notes
    -----
    : (1) slice the input array by the classification fields
    : (2) sort the sliced array using the flds as sorting keys
    : (3) use unique on the sorted array to return the results
    : (4) a quick histogram to get the counts until numpy 1.12 can be used
    : ... then ship the results back.  only uni and vals is needed. The
    :     rest is for testing and future work.
    >>> np.unique(ar, return_index=False, return_inverse=False,
    ...           return_counts=False, axis=None)
    """
    a = a[flds]  # (1)
    idx = np.argsort(a, axis=0, order=flds)  # (2)
    a_sort = a[idx]
    uni, cnts = np.unique(a_sort, return_counts=True)  # (3)
    out = rfn.append_fields(uni, names="Counts", data=cnts, usemask=False)
    return out


if len(sys.argv) == 1:
    testing = True
    in_tbl = r"C:\Arc_projects\Table_tools\Table_tools.gdb\SamplingGrids"
    all_flds = "*"
    cls_flds = 'Row_class;Col_class'
else:
    testing = False
    in_tbl = sys.argv[1]           # input table
    out_tbl = sys.argv[2]          # results table
    cls_flds = sys.argv[3]         # classification field(s)

# ---- common tasks
cls_flds = cls_flds.split(";")  # tidy up the two field inputs
a = arcpy.da.TableToNumPyArray(in_tbl, "*")  # use the full array's data

out = freq(a, cls_flds)  # do freq analysis

# create the output array and return the table
if not testing:
    arcpy.da.NumPyArrayToTable(out, out_tbl)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Do nothing for now.
    """

