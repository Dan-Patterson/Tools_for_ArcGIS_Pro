# -*- coding: UTF-8 -*-
"""
:Script:   numpyarray2raster.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-12
:Purpose:  tools for working with numpy arrays
:Useage:
:
: NumPyArrayToRaster(in_array, {lower_left_corner},
:                              {x_cell_size}, {y_cell_size},
:                              {value_to_nodata})
:References:
:  http://pro.arcgis.com/en/pro-app/arcpy/functions/
:       numpyarraytoraster-function.htm

:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import os
from textwrap import dedent
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['arr_tif_tf',    # array to tiff using tifffile
           'arr_tif_PIL',   # .... using PIL
           'arr_tif_cb',    # .... using CompositeBands
           'tifffile_arr'
           ]


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


def arr_tif_tf(a, fname):
    """Convert a NumPy array to a tiff using tifffile.py
    :
    : - https://github.com/blink1073/tifffile  source github page
    :Requires:  tifffile
    :---------
    : from tifffile import imread, imsave, TiffFile
    : help(TiffFile.geotiff_metadata)
    :
    : imsave:
    : imsave(file, data=None, shape=None, dtype=None, bigsize=2**32-2**25,
    :       **kwargs)
    :  file - filename with *.tif
    :  data - array
    :  shape - if creating an empty array
    : a = np.arange(4*5*6).reshape(4, 5, 6)
    : imsave('c:/temp/temp.tif', a)
    : b =imread('c:/temp/temp.tif')
    : np.all(a == b)  # True
    :
    : GeoTiff, World files:
    :  http://trac.osgeo.org/geotiff/
    :  https://en.wikipedia.org/wiki/World_file
    """
    import warnings
    warnings.filterwarnings('ignore')
    from tifffile import imsave
    imsave(fname, a)


def arr_tif_PIL(a, fname):
    """convert an array to a tif using PIL
    """
    from PIL import Image
    imgs = []
    for i in a:
        imgs.append(Image.fromarray(i))
    imgs[0].save(fname, compression="tiff_deflate", save_all=True,
                 append_images=imgs[1:])
    # ---- done


def arr_tif_cb(a, fname, LL_X=0, LL_Y=0, cell_size=1, no_data=None):
    """Array to tiff using esri compositebands
    :
    """
    #  This section works
    arcpy.env.workspace = 'in_memory'
    pnt = arcpy.Point(LL_X, LL_Y)
    rasters = []
    if no_data is None:
        no_data = np.iinfo(a.dtype.type).min
    for i in range(a.shape[0]):
        ai = a[i]
        rast = arcpy.NumPyArrayToRaster(ai, pnt, cell_size, cell_size, no_data)
        r_name = "in_memory/a{:0>3}.tif".format(i)
        rasters.append(r_name)
        rast.save(r_name)
    rasters = ";".join([i for i in rasters])
    # Mosaic dataset section
    arcpy.management.CompositeBands(rasters, fname)
    # ----


def tifffile_arr(fname):
    """Convert tif to array using tifffile
    :
    : Source: tifffile # http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
    :-------
    : TiffFile.asarray(self, key=None, series=None, out=None, maxworkers=1)
    :  key - int, slice, or sequence of page indices
    :        Defines which pages to return as array.
    :  series - int or TiffPageSeries
    :  out - array if None or filename
    :  maxworkers = default 1, number of threads to use to get data
    #
    : from tifffile import tifffile as tf ******
    :
    : tif = TiffFile(fp)  # *****
    : tif.byteorder  # '<'
    : tif.isnative   # True
    """
    from tifffile import TiffFile
    with TiffFile(fname) as tif:
        a = tif.asarray()
        a = np.rollaxis(a,  axis=2, start=0)
    return a


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    """
    a = np. array(['1, 2, 3, 4, 5', 'a, b, c', '6, 7, 8, 9',
                   'd, e, f, g, h', '10, 11, 12, 13'])
    return a


msg0 = """
: -----------------------------------------------------------
Script....
.... {}
Failed because either...

(1) Input raster needs to be an *.npy file.
(2) The input and/or output path and/or filename has a space in it.
(3) The output raster has to be a *.tif

Fix one or more conditions...

...
: -----------------------------------------------------------
"""

msg1 = """
: -----------------------------------------------------------
Script....
.... {}
Completed....
...
: -----------------------------------------------------------
"""


def check_files(file_path, ext=""):
    """Check expected file paths and extensions, to ensure compliance with
    :  tool specifications
    """
    is_good = True
    head, tail = os.path.split(file_path)
    if not os.path.exists(head):
        return False
    if " " in tail:
        return False
    if os.path.splitext(tail)[1] != ext:
        return False
    if " " in file_path:
        return False
    return is_good
    # ----


def check_nodata(a, nodata):
    """check for array dtype etc"""
    a_kind = a.dtype.str
    if a_kind in ('|i1', '<i2', '<i4', '<i8'):
        nodata = int(nodata)
    elif a_kind in ('<f2', '<f4', '<f8'):
        nodata = float(nodata)
    else:
        nodata = -9999
    return nodata


def _tool():
    """run when script is from a tool
    """
    in_npy = sys.argv[1]
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    cell_size = float(sys.argv[4])
    no_data = sys.argv[5]
    out_rast = sys.argv[6]
    SR = sys.argv[7]
    msg = """
    : -----------------------------------------------------------
    Script parameters....
    in_npy {}
    LL corner (x,y) ... {}, {}
    cell size ... {}
    nodata ...... {}
    out raster .. {}
    : -----------------------------------------------------------
    """
    args = [in_npy, x, y, cell_size, no_data, out_rast]
    tweet(dedent(msg).format(*args))
    #
    # ---- main tool section
    pnt = arcpy.Point(X=x, Y=y)
    is_good1 = check_files(in_npy, ext=".npy")
    is_good2 = check_files(out_rast, ext=".tif")
    if is_good1 and is_good2:
        a = np.load(in_npy)
        no_data = check_nodata(a, no_data)  # check nodata value
        rast = arcpy.NumPyArrayToRaster(a, pnt, cell_size, cell_size, no_data)
        rast.save(out_rast)
        if SR not in ('#', ' ', '', None, 'Unknown'):
            arcpy.DefineProjection_management(out_rast, SR)
        tweet(dedent(msg1).format(script))
    else:
        raise AttributeError(msg0)
        tweet(dedent(msg0).format(script))
    # ----


# ----------------------------------------------------------------------
# .... final code section
if len(sys.argv) == 1:
    testing = True
    a = _demo()
    frmt = "Result...\n{}"
    print(frmt.format(a))
else:
    testing = False
    _tool()
#
if not testing:
    print('Done...')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    a = _demo()
