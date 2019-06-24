# -*- coding: UTF-8 -*-
"""
=========
frequency
=========

Script :
    frequency.py

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-06-22
    Original, 2018

Purpose :
    To supplant the Frequency tool for those that don't have an
    advanced license.  Performs a frequency count of unique combinations of
    fields and a set of statistical summary parameters.

Useage :
    Load the toolbox into Pro and run the tool script from there.

References
----------
`Frequency
<https://pro.arcgis.com/en/pro-app/tool-reference/analysis/frequency.htm>`_.
Link to ArcGIS Pro help files current as of modified date.

Notes
-----

>>> to_array = arcpy.da.TableToNumPyArray(r"C:\folder\sample.dbf", "*")
>>> arcpy.da.NumPyArrayToTable(from_array, r"C:\folder_tbl\test.gdb\out")

:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import numpy.lib.recfunctions as rfn
import arcpy
arcpy.env.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ============================================================================
# ---- frequency section
#
msg0 = """
Either you failed to specify the geodatabase location and filename properly
or you had flotsam, including spaces, in the path, like...\n
  {}\n
Create a safe path and try again...\n
`Filenames and paths in Python`
https://community.esri.com/blogs/dan_patterson/2016/08/14/filenames-and
-file-paths-in-python.
"""
def check_path(out_fc):
    """Check for a filegeodatabase and a filename"""
    _punc_ = '!"#$%&\'()*+,-;<=>?@[]^`{|}~ '
    flotsam = " ".join([i for i in _punc_]) + " ... plus the `space`"
    msg = msg0.format(flotsam)
    if np.any([i in out_fc for i in _punc_]):
        return (None, msg)
    pth = out_fc.split("\\")
    if len(pth) == 1:
        return (None, msg)
    name = pth[-1]
    gdb = "\\".join(pth[:-1])
    if gdb[-4:] != '.gdb':
        return (None, msg)
    return gdb, name


def freq(a, cls_flds=None, stat_fld=None):
    """Frequency and crosstabulation

    Parameters
    ----------
    a : array
        A structured array
    flds : field
        Fields to use in the analysis

    Notes
    -----
    1. slice the input array by the classification fields
    2. sort the sliced array using the flds as sorting keys
    3. use unique on the sorted array to return the results and the counts

    >>> np.unique(ar, return_index=False, return_inverse=False,
    ...           return_counts=True, axis=None)
    """
    if stat_fld is None:
        a = a[cls_flds]  # (1) It is actually faster to slice the whole table
    else:
        all_flds = cls_flds + [stat_fld]
        a = a[all_flds]
    idx = np.argsort(a, axis=0, order=cls_flds)  # (2)
    a_sort = a[idx]
    uni, inv, cnts = np.unique(a_sort[cls_flds], False,
                               True, return_counts=True)  # (3)
    out_flds = "Counts"
    out_data = cnts
    if stat_fld is not None:
        splitter = np.where(np.diff(inv) == 1)[0] + 1
        a0 = a_sort[stat_fld]
        splits = np.split(a0, splitter)
        sums = np.asarray([np.nansum(i.tolist()) for i in splits])
        nans = np.asarray([np.sum(np.isnan(i.tolist())) for i in splits])
        mins = np.asarray([np.nanmin(i.tolist()) for i in splits])
        means = np.asarray([np.nanmean(i.tolist()) for i in splits])
        maxs = np.asarray([np.nanmax(i.tolist()) for i in splits])
        out_flds = [out_flds, stat_fld + "_sums", stat_fld + "_NaN",
                    stat_fld + "_min", stat_fld + "_mean", stat_fld + "_max"]
        out_data = [out_data, sums, nans, mins, means, maxs]
    out = rfn.append_fields(uni, names=out_flds, data=out_data, usemask=False)
    return out

# ---- testing and tool section ----------------------------------------------
#
if len(sys.argv) == 1:
    testing = True
    in_tbl = r"C:\Arc_projects\Free_Tools\Free_tools.gdb\sample_10k"
    all_flds = "*"
    cls_flds = 'County;Town_class'  # Facility, Time
    stat_fld = 'Age'
else:
    testing = False
    in_tbl = sys.argv[1]    # input table
    cls_flds = sys.argv[2]  # classification field(s)
    stat_fld = sys.argv[3]
    out_tbl = sys.argv[4]   # results table

# ---- common tasks
cls_flds = cls_flds.split(";")  # multiple to list, make a singleton a list
if stat_fld in (None, 'NoneType', ""):
    stat_fld = None
a = arcpy.da.TableToNumPyArray(in_tbl, "*")  # use the whole array
out = freq(a, cls_flds, stat_fld)  # do freq analysis

# ---- create the output array and return the table
# ----
if not testing:
    result = check_path(out_tbl)
    if result[0] is None:
        msg = "...\n{}\n...".format(result[1])
        print(msg)
        arcpy.AddMessage(msg)
    else:
        gdb, name = result
    arcpy.da.NumPyArrayToTable(out, out_tbl)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Do nothing for now.
    """

