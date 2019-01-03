# -*- coding: utf-8 -*-
"""
ras2npy.py
==========

Script :   ras2npy.py  # raster to numpy array as *.npy file

Author :   Dan_Patterson@carleton.ca

Modified : 2018-09-25

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/functions/rastertonumpyarray
-function.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import os

import numpy as np
from art_common import (tweet, rasterfile_info)
from arcpy import Point, Raster, RasterToNumPyArray, env

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

env.overwriteOutput = True

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    pth = script.split("/")[:-2]
    pth0 = "/".join(pth) + "/Data/r00.tif"
    r = Raster(pth0)
    out_arr = "/".join(pth) + "/Data/r01.npy"
    frmt = "Result...\n{}"
#    print(frmt.format(a))
else:
    testing = False
    pth = sys.argv[1]
    out_arr = sys.argv[2]
    r = Raster(pth)
# parameters here
LL = r.extent.lowerLeft
cols = int(r.extent.width/r.meanCellWidth)
rows = int(r.extent.height/r.meanCellWidth)
a = RasterToNumPyArray(r,
                       lower_left_corner=Point(LL.X, LL.Y),
                       ncols=cols,
                       nrows=rows,
                       nodata_to_value=r.noDataValue
                       )
#
# ---- overwrite existing outputs
if os.path.isfile(out_arr):
    tweet("\nRemoving ... {}\nbefore saving".format(out_arr))
    os.remove(out_arr)
np.save(out_arr, a)
if testing:
    tweet('\nScript source... {}'.format(script))
print('\nCleaning up')
del r, tweet, rasterfile_info, Point, Raster, RasterToNumPyArray
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
