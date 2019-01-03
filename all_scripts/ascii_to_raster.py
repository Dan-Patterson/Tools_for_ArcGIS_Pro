# coding: utf-8
"""
ascii_to_raster
===============

Script :   ascii_to_raster.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-11-20

Purpose :  Convert an ascii file to a raster (tif)



syntax:
    RasterToNumPyArray(in_raster, {lower_left_corner},
                      {ncols}, {nrows}, {nodata_to_value})

Useage:
-------
inRas : arcpy.Raster('C:/data/inRaster')

lowerLeft : arcpy.Point(inRas.extent.XMin,inRas.extent.YMin)

cellSize : ras.meanCellWidth

Returns:
--------
bands, rows, columns or a structured array

Notes:
------
- rows is dim 0
- cols is dim 1
- depth is dim 2

References:
-----------
`<http://desktop.arcgis.com/en/arcmap/latest/analyze/arcpy-functions/
rastertonumpyarray-function.htm>`_.

From my post : 2011-10-11

`<http://gis.stackexchange.com/questions/16098/determining-min-and->`_.
max-values-in-an-ascii-raster-dataset-using-python/16101#16101

>>> import numpy as np
>>> ascii_file = "c:/temp/Ascii_3x3_1nodata.asc"
>>> an_array = np.mafromtxt(ascii_file, 'float', '#', None, 6, None, '-999')
NCOLS          3
NROWS          3
XLLCORNER      0
YLLCORNER      0
CELLSIZE       1
NODATA_VALUE   -999
0 1 2
-999 4 5
6 7 8

"""
import os
import numpy as np
from textwrap import dedent, indent
import arcpy

arcpy.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=80, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

# ---- Processing temporal ascii files ----
# Header information
# ncols 720
# nrows 360
# xllcorner -180
# yllcorner -90
# cellsize 0.5
# NODATA_Value -9999
# ---- Begin the process ----
#
# =============================================================================
# save_output = False
# cols = 720
# rows = 360
# ll_corner =  arcpy.Point(-180., -90.0)  # to position the bottom left corner
# dx_dy = 0.5
# nodata = '-9999'
# #
# # ---- create the basic workspace parameters, create masked arrays ----
# #
# out_file = r'c:\Data\ascii_samples\avg_yr.tif'
# folder = r'C:\Data\ascii_samples'
# arcpy.env.workspace = folder
# ascii_files = arcpy.ListFiles("*.asc")
# a_s = [folder + '\{}'.format(i) for i in ascii_files]
# arrays = []
#
# for arr in a_s[:1]:
#     a = np.mafromtxt(arr, dtype='int32', comments='#',
#                      delimiter=' ', skip_header=6,
#                      missing_values=nodata, usemask=True)
#     value_to_nodata = int(a.get_fill_value())
#     out = a.view(np.ndarray, fill_value=value_to_nodata)
#     r = arcpy.NumPyArrayToRaster(out, ll_corner, dx_dy, dx_dy)
#     out_file = arr.replace(".asc", ".tif")
#     if save_output:
#         r.save(out_file)
#     del r
# =============================================================================
from arcpy import NumPyArrayToRaster, Point

path = r"C:\Temp\dem.txt"
ncols    =     317
nrows     =    204
xllcorner =    2697732
yllcorner =    1210264
cellsize  =    2
NODATA_value = -9999

a = np.genfromtxt(path, np.float, delimiter=' ', skip_header=6)
a0 = np.where(a==-9999., np.nan, a)

LL = Point(xllcorner, yllcorner)
out = NumPyArrayToRaster(a0, LL, 2.0, 2.0, np.nan)
# out.save(r"C:\Temp\dem_np.tif")
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """

#    print("Script... {}".format(script))
#    _demo()
