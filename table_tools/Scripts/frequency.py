# -*- coding: UTF-8 -*-
"""
:Script:   frequency.py
:Author:   Dan_Patterson@carleton.ca
:Modified: 2017-04-14
:Purpose:  To supplant the Frequency tool for those that don't have an
:   advanced license.
:Useage:  Load the toolbox into Pro and run the tool script from there.
:Reference:
:    http://desktop.arcgis.com/en/arcmap/latest/tools/
:         analysis-toolbox/frequency.htm
:Notes:
: - to_array = arcpy.da.TableToNumPyArray(r"C:\folder\sample.dbf", "*")
: - arcpy.da.NumPyArrayToTable(from_array, r"C:\folder_tbl\test.gdb\out")
:
:Dev Info:
:  tbx - C:\GIS\Tools_scripts\Table_tools\Table_tools.tbx
:  script - C:\GIS\Tools_scripts\Table_tools\Scripts\frequency.py
:  arcpy.Tabletools.Frequency("polygon_demo",
:             r"C:\GIS\Tools_scripts\Table_tools\Table_tools.gdb\f2",
:             "Test;main_part", None)
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
    : a - input structured array
    : flds - fields to use in the analysis
    :Notes:
    :-----
    : (1) slice the input array by the classification fields
    : (2) sort the sliced array using the flds as sorting keys
    : (3) use unique on the sorted array to return the results
    : (4) a quick histogram to get the counts until numpy 1.12 can be used
    : ... then ship the results back.  only uni and vals is needed. The
    :     rest is for testing and future work.
    """
    a = a[flds]  # (1)
    idx = np.argsort(a, axis=0, order=flds)  # (2)
    a_sort = a[idx]
    final = np.unique(a_sort, return_index=True, return_inverse=True)  # (3)
    uni = final[0]
    first = final[1]
    cls = final[2]
    cases = np.arange(len(uni)).tolist()
    cases.append(cases[-1] + 1)
    count = np.histogram(cls, cases)  # (4)
    dt = (uni.dtype.descr)
    dt.append(('count', '<i4'))
    vals = count[0]
    return uni, first, cls, cases, count, vals


def summ(a, cls_flds, uniq, sum_flds):
    """sum the input field
    : a is the large array sliced by the classification fields
    : uniq - unique classes
    :
    """
    to_sum = a[cls_flds]
    out_sum = []
    for cl in uniq:
        rows = a[to_sum == cl]
        out_sum.append(np.sum(rows[sum_flds]))
    return out_sum


in_tbl = sys.argv[1]           # input table
out_tbl = sys.argv[2]          # results table
cls_flds = sys.argv[3]         # classification field(s)
sum_flds = sys.argv[4]         # fields for doing sums on

cls_flds = cls_flds.split(";")  # tidy up the two field inputs
sum_flds = sum_flds.split(";")

a = arcpy.da.TableToNumPyArray(in_tbl, "*")  # use the full array's data

uni, first, cls, cases, count, vals = freq(a, cls_flds)  # do freq analysis

# perform the summary results
new_vals = [vals]
new_names = ['count']
arcpy.AddMessage("sum flds = {}".format(sum_flds))

if sum_flds[0] != '#':
    for i in sum_flds:
        fld_sums = summ(a, cls_flds, uni, i)  # do the sums using summ
        new_names.append('sum_' + i)
        new_vals.append(fld_sums)

# create the output array and return the table
b = rfn.append_fields(uni, names=new_names, data=new_vals, usemask=False)

arcpy.da.NumPyArrayToTable(b, out_tbl)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Do nothing for now.
    """
#    in_tbl = r"C:\GIS\Tools_scripts\Table_tools\Table_tools.gdb\polygon_demo"
#    all_flds = "*"
#    cls_flds = 'Test;main_part'
#    sum_flds = 'Shape_Area;Shape_Length'
