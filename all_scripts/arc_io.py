# -*- coding: UTF-8 -*-
"""
arc_io
======

Script :   arc_io.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-09-14

Purpose : Basic io tools for numpy arrays and arcpy

Notes :
::
    1.  load_npy    - load numpy npy files
    2.  save_npy    - save array to *.npy format
    3.  read_txt    - read array created by save_txtt
    4.  save_txt    - save array to npy format
    5.  arr_json    - save to json format
    6.  array2raster - save array to raster
    7.  rasters2nparray - batch rasters to numpy array

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
import os
import numpy as np
import arcpy


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['array2raster',
           'rasters2nparray',
           ]

# ----------------------------------------------------------------------
# (1) batch load and save to/from arrays and rasters
def array2raster(a, folder, fname, LL_corner, cellsize):
    """It is easier if you have a raster to derive the values from.

    >>> # Get one of the original rasters since they will have the same
    >>> # extent and cell size needed to produce output
    >>> r01 = rasters[1]
    >>> rast = arcpy.Raster(r01)
    >>> lower_left = rast.extent.lowerLeft
    >>> # this is a Point object... ie LL = arcpy.Point(10, 10)
    >>> cell_size = rast.meanCellHeight  # --- we will use this for x and y
    >>> f = r'c:/temp/result.tif'  # --- don't forget the extention

    Requires:
    ---------

    `arcpy` and `os` if not previously imported
    """
    if not os.path.exists(folder):
        return None
    r = arcpy.NumPyArrayToRaster(a, LL_corner, cellsize, cellsize)
    f = os.path.join(folder, fname)
    r.save(f)
    print("Array saved to...{}".format(f))


# ----------------------------------------------------------------------
# (7) batch load and save to/from arrays and rasters
def rasters2nparray(folder=None, to3D=False):
    """Batch the RasterToNumPyArray arcpy function to produce 3D or a list
    of 2D arrays

    NOTE:
    ----
    Edit the code... far simpler than accounting for everything.
    There is a reasonable expectation that rasters exist in the folder.

    Requires:
    --------
    modules :
        os, arcpy if not already loaded
    folder : folder
        A folder on disk... a real one
    to3D : boolean
        If False, a list of arrays, if True a 3D array
    """
    arrs = []
    if folder is None:
        return None
    if not os.path.exists(folder):
        return None
    arcpy.env.workspace = folder
    rasters = arcpy.ListRasters("*", "TIF")
    for raster in rasters:
        arrs.append(arcpy.RasterToNumPyArray(raster))
    if to3D:
        return np.array(arrs)
    else:
        return arrs


def _demo():
    """
    : -
    """
    _npy = "/Data/sample_20.npy"  # change to one in the Data folder
    _npy = "{}".format(script.replace('/fc_tools/arc_io.py',_npy))
    return _npy


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    fname = _demo()
