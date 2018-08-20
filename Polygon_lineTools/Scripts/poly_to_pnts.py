# -*- coding: utf-8 -*-
"""
poly_to_pnts.py
===============

Script :   poly_to_pnts.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-08-05

Purpose:
--------
    Tools for working with numpy arrays.  This converts poly* features
    to points

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""

import sys
import numpy as np
from arcpytools_plt import fc_array, fc_info, tweet  #, frmt_rec, _col_format
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

def append_fld(arr, name='Pnt_id', data=None):
    """Append an id field to a subarray of points.
    Emulated after recfunctions append_fields
    """
    dt = arr.dtype.descr + [(name, '<i4')]
    if data is None:
        data = np.arange(arr.shape[0])
    e = np.empty(arr.shape[0], dtype=dt)
    dt_new = np.dtype(dt)
    for i in dt_new.names[:-1]:
        e[i] = arr[i]
    e[dt_new.names[-1]] = data
    return e


def to_pnts(in_fc, out_fc, keep_flds=None, to_file=False):
    """Convert a featureclass to a point file with unique point id values
    added to indicate the points within the poly* features.  The output field
    names (keep_flds) can be specified or all will be used if '*' is used.

    Requires:
    ---------
        fc_array - from arcpytools_plt.py
    Notes:
    ------
    - a Pnt_id field is added to indicate the order of the points making the
      poly* feature
    - the duplicate last point is removed for polygon features
    - potentially troublesome field names are changed.
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    if keep_flds is None:
        keep_flds = "*"
    a, out_flds, SR = fc_array(in_fc, flds=keep_flds, allpnts=True)
    a_s = np.split(a, np.where(np.diff(a[oid_fld]))[0] + 1)
    out = []
    for i in a_s:
        ids = np.arange(i.shape[0])
        out.append(append_fld(i, name='Pnt_id', data=ids ))
    # ---- remove duplicate last point and stack the points
    if shp_type == "Polygon":
        for i in range(len(a_s)):
            out[i] = out[i][:-1]
    out = np.hstack(out)
    # ---- replace dodgy field names
    kv = {'OBJECTID': 'Old_ID', 'SHAPE@X': 'X_', 'SHAPE@Y': 'Y_',
          'Shape_Length': 'Leng_orig', 'Shape_Area': 'Area_orig'}
    change = [kv.get(i, i) for i in out.dtype.names]
    out.dtype.names = change
    if to_file:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.da.NumPyArrayToFeatureClass(out, out_fc,
                                          ['X_', 'Y_'], SR)
    return out


# ---- Do the work
#
if len(sys.argv) == 1:
    testing = True
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    in_fc = "/".join(in_pth) + "/shapes_mtm9"
    out_fc = "/".join(in_pth) + "/shape_pnts2"
    keep_flds = "*"
    out = to_pnts(in_fc, out_fc, keep_flds, to_file=False)
else:
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]
    keep_flds = sys.argv[3]
    if keep_flds not in ("#", "", " ", None):
        keep_flds = keep_flds.split(";")
    tweet(keep_flds)
    out = to_pnts(in_fc, out_fc, keep_flds, to_file=True)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """