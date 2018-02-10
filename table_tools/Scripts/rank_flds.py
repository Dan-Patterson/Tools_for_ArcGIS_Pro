# -*- coding: UTF-8 -*-
"""
:Script:   rank_flds.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-11-01
:Purpose:  tools for working with numpy arrays
:Useage:  Runk the toolbox, select the parameters
:
:References: sure are
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

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
for v in vals:
    fix_null = np.where(arr[v] == 'None', '', arr[v])
    arr[v] = fix_null
arr_sort = np.sort(arr, order=in_flds)
dt = [(oid, '<i4'), (out_fld, '<i4')]
out_array = np.zeros((arr_sort.shape[0],), dtype=dt)
dt_names = out_array.dtype.names
out_array[dt_names[0]] = arr_sort[oid]
out_array[dt_names[-1]] = np.arange(1, arr_sort.size + 1)  # gdb tables
arcpy.da.ExtendTable(in_tbl, 'OBJECTID', out_array, 'OBJECTID')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
