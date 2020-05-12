# -*- coding: utf-8 -*-
"""
:Script:   sortpnts.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-02-28
:Purpose: Sort points by X or Y in ascending or descending order
"""

import sys
import numpy as np
import arcpy
from arcpytools_pnt import fc_info, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=100, precision=2,
                    suppress=True, threshold=120, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

# ---- Convert the featureclass to points returning base information ----
# ---- The 'Shape' field is changed to X and Y to facilitate sorting etc.
in_fc = sys.argv[1]
srt_order = sys.argv[2]
ascend = sys.argv[3]
out_fc = sys.argv[4]

shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)

a = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR)
dt = [('X', '<f8'), ('Y', '<f8')]
shps = np.array([tuple(i) for i in a[shp_fld]], dtype=dt)

if srt_order == 'X':
    idx = np.argsort(shps, order=('X', 'Y'))
else:
    idx = np.argsort(shps, order=('Y', 'X'))

shps = a[idx]
if not ascend:
    shps = shps[::-1]

arcpy.da.NumPyArrayToFeatureClass(shps, out_fc, shp_fld, SR)
#
frmt = """\n\nScript.... {}\nUsing..... {}\nSR...{}\nSorting by... {},
ascending... {}\nProducing ... {}\n"""
args = [script, in_fc, SR.name, srt_order, ascend, out_fc]
tweet(frmt.format(*args))

# -------------------------------------------------------------------------
if __name__ == "__main__":
    """ No demo  """
#    in_fc = r"C:\GIS\Geometry_projects\Spiral_sort\Polygons\Parcels.shp"
