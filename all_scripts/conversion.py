# -*- coding: UTF-8 -*-
"""
conversion
==========

Script :   conversion.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-04-18

Purpose :  conversion tools for working with numpy arrays

Functions
---------

>>> __all__ = ['arr2tuple',
               'recarray2dict',
               'table2dict',
               'read_npy',
               'save_as_txt',
               'arr_tif_tf',    # array to tiff using tifffile
               'arr_tif_cb',    # .... using CompositeBands
               'tifffile_arr',  # tiff to array using tifffile
               '_demo_tif'
               ]
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import warnings
from textwrap import dedent
import numpy as np
from numpy.lib import format
#import tifffile
#from tifffile import imread, TiffFile
import arcpy

warnings.filterwarnings('ignore')

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=200, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['arr2tuple',
           'recarray2dict',
           'table2dict',
           'read_npy',
           'save_as_txt',
           'arr_tif_tf',    # array to tiff using tifffile
           'arr_tif_cb',    # .... using CompositeBands
           'tifffile_arr',  # tiff to array using tifffile
           '_demo_tif'
           ]

# ---- general functions -----------------------------------------------------
#
def arr2tuple(a):
    """Convert an array to tuples, a convenience function when array.tolist()
    doesn't provide the necessary structure since lists and lists of lists
    aren't hashable
    """
    return tuple(map(tuple, a.tolist()))


# ---- array or table to dictionary ------------------------------------------
#
def recarray2dict(a, flds=None, to_list=False):
    """Return a numpy.recarray as dictionary

    a :
        a NumPy structured or recarray with/without mixed dtypes
    to_list :
        if True, the dictionary values are arrays, if False, the values
        are returned as lists
    """
    if flds is None:
        flds = a.dtype.names
    if to_list:
        tbl_dct = {nme: a[nme].tolist() for nme in flds}
    else:
        tbl_dct = {nme: a[nme] for nme in flds}
    return tbl_dct


def table2dict(in_tbl, flds=None, to_list=True):
    """Searchcursor to dictionary

    in_tbl : table
        a geodatabase table
    flds : list or tuple
        a list/tuple of field names, if None, then all the fields are used
    to_list : boolean
        True, then a list of the field values is returned.

        False, an array subset from the array 'a' is returned.
    """
    if flds is None:
        flds = [f.name for f in arcpy.ListFields(in_tbl)]
    elif not isinstance(flds, (list, tuple)):
        flds = [flds]
    a = arcpy.da.TableToNumPyArray(in_tbl, flds)
    if to_list:
        tbl_dct = {nme: a[nme].tolist() for nme in flds}
    else:
        tbl_dct = {nme: a[nme] for nme in flds}
    return tbl_dct


# ---- array to and from npy ------------------------------------------------
#
def read_npy(fp, prn=False):
    """ Read an npy file quickly

    fp : string
        The file path: "c:/temp/a01.npy"
    prn : boolean
        obtain full information if True

    Requires:
    ---------
    from numpy.lib import format

    Notes:
    -------
    shortcut ... np.load("c:/temp/a01.npy")
    """
    frmt = """
    ---- npy reader ---------------------------------------------------------
    File  {}
    Shape {},  C-contig {},  dtype {}
    Magic {}
    -------------------------------------------------------------------------
    """
    with open(fp, 'rb') as f:
        major, minor = format.read_magic(f)
        mag = format.magic(major, minor)
        shp, is_fortran, dt = format.read_array_header_1_0(f)
        count = np.multiply.reduce(shp, dtype=np.int64)
        BUFFER_SIZE = 2**18
        max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dt.itemsize)
        array = np.ndarray(count, dtype=dt)
        for i in range(0, count, max_read_count):
            cnt = min(max_read_count, count - i)
            read_size = int(cnt * dt.itemsize)
            data = format._read_bytes(f, read_size, "array data")
            array[i:i + cnt] = np.frombuffer(data, dtype=dt, count=cnt)
        array.shape = shp
    if prn:
        print(dedent(frmt).format(fp, shp, (not is_fortran), dt, mag))
    return array


def save_as_txt(fname, a):
    """Save a numpy array as a text file determining the format from the
    data type.

    Reference:
    ----------
    from numpy savetxt

    >>> savetxt(fname, X, fmt='%.18e', delimiter=' ',
                newline=`\\n`, header='', footer='', comments='# ')

    - fmt : '%[flag]width[.precision]specifier'
    - fmt='%.18e'
    """
    dt_kind = a.dtype.kind
    l_sze = max(len(str(a.max())), len(str(a.min())))
    frmt = '%{}{}'.format(l_sze, dt_kind)
    hdr = "dtype: {} shape: {}".format(a.dtype.str, a.shape)
    np.savetxt(fname, a, fmt=frmt, delimiter=' ',
               newline='\n', header=hdr, footer='', comments='# ')


# ---- array to and from tif files ------------------------------------------
#
def arr_tif_tf(a, fname):
    """Convert a NumPy array to a tiff using tifffile.py

    Requires:
    ---------
    from tifffile import imread, imsave, TiffFile

    help(TiffFile.geotiff_metadata)

    imsave :
        imsave(file, data=None, shape=None, dtype=None, bigsize=2**32-2**25,
              **kwargs)

    file - filename with *.tif

    data - array

    shape - if creating an empty array

    >>> a = np.arange(4*5*6).reshape(4, 5, 6)
    >>> imsave('c:/temp/temp.tif', a)
    >>> b =imread('c:/temp/temp.tif')
    >>> np.all(a == b)  # True

    GeoTiff, World files:
    ---------------------
    [1]
    https://github.com/blink1073/tifffile  source github page
    [2]
    http://trac.osgeo.org/geotiff/
    [3]
    https://en.wikipedia.org/wiki/World_file

    imsave('temp.tif', data, compress=6, metadata={'axes': 'TZCYX'})

    Parameters ‘append’, ‘byteorder’, ‘bigtiff’, ‘software’, and ‘imagej’,
    are passed to the TiffWriter class.

    Parameters ‘photometric’, ‘planarconfig’, ‘resolution’, ‘compress’,
    ‘colormap’, ‘tile’, ‘description’, ‘datetime’, ‘metadata’, ‘contiguous’
    and ‘extratags’ are passed to the TiffWriter.save function.
    """
    import warnings
    warnings.filterwarnings('ignore')
    from tifffile import imsave
    imsave(fname, a)


def arr_tif_cb(a, fname, LL_X=0, LL_Y=0, cell_size=1, no_data=None):
    """Array to tif using esri compositebands
    : a - array, at least 2D
    : fname - full filename of tif to save
    : LL_X, Y - coordinate of the lower left in real world coordinates
    : cell_size - guess... things only make sense using projected coordinates
    : no_data - specify the value if any, one will be assigned if None
    """
    arcpy.env.workspace = 'in_memory'
    pnt = arcpy.Point(LL_X, LL_Y)
    rasters = []
    if no_data is None:
        if a.dtype.kind == 'i':
            no_data = np.iinfo(a.dtype.type).min
        elif a.dtype.kind == 'f':
            no_data = np.finfo(a.dtype.type).min
        else:
            no_data = a.min() - 1
    for i in range(a.shape[0]):
        ai = a[i]
        rast = arcpy.NumPyArrayToRaster(ai, pnt, cell_size, cell_size, no_data)
        r_name = "in_memory/a{:0>3}.tif".format(i)
        rasters.append(r_name)
        rast.save(r_name)
    rasters = ";".join([i for i in rasters])
    #
    arcpy.management.CompositeBands(rasters, fname)
    # ----
    return


def tifffile_arr(fname, verbose=False):
    """Convert tif to array using tifffile
    :
    :Source: tifffile # http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
    :Requires: from tifffile import imread, TiffFile
    :  fname - full tiff file path and filename
    :  verbose - True to print all information gleaned
    :Useage:
    :(1) a = imread(fname) ... or
    :(2) TiffFile.asarray(self, key=None, series=None, out=None, maxworkers=1)
    :    key - int, slice, or sequence of page indices
    :          Defines which pages to return as array.
    :    series - int or TiffPageSeries
    :    out - array if None or filename
    :    maxworkers = default 1, number of threads to use to get data
    :Extra info:
    : kys = tif.__dict__.keys()
    :     for k in kys:
    :         print("{!s:<15}: {!s:<30}".format(k, tif.__dict__[k]))
    """
    with TiffFile(fname) as tif:
        if verbose:
            print("\nTiff file: {}\nflags: {}".format(fname, tif.flags))
        if tif.is_shaped and verbose:
            print("Shape info: {}\n".format(tif.shaped_metadata))
        if tif.is_geotiff and verbose:
            print("Geotiff info:")
            d = tif.geotiff_metadata
            for key in d.keys():
                print("- {}:   {}".format(key, d[key]))
        #
        a = tif.asarray()
        #
        if tif.is_tiled:
            a = np.rollaxis(a, axis=2, start=0)
    return a, tif  # uncomment and return tif for testing


def rast_np(fp):
    """shortcut implementation to RasterToNumPyArray
    : fp - file path and name
    : LL_X, LL_Y - lower left coordinates
    """
    a = arcpy.RasterToNumPyArray(fp)
    return a


def _demo_tif():
    """Code to run if in demo mode
    """
#    a = np.arange(5*3*4, dtype=np.int8).reshape(5, 3, 4)  # int8
#    a = np.arange(5*3*4, dtype=np.int16).reshape(5, 3, 4)  # int16
#    a = np.arange(5*3*4, dtype=np.int32).reshape(5, 3, 4)  # int32
#    a = np.arange(5*3*4, dtype=np.int64).reshape(5, 3, 4)  # int64
#    a = np.arange(5*3*4, dtype=np.float16).reshape(5, 3, 4)  # float16
#    a = np.arange(5*3*4, dtype=np.float32).reshape(5, 3, 4)  # float32
    a = np.arange(5*3*4, dtype=np.float64).reshape(5, 3, 4)  # float64
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo or uncomment other sections
    """
#    print("Script... {}".format(script))
#    a = _demo_tif()
    # ---- tifffile section
#    f = "/Data/a30_tifffile.tif"
#    f = "/Data/rast_composite_9bands.tif"
#    pth = script.split("/")[:-2]
#    fname = "/".join(pth) + f
    # ---- table section
#    in_tbl = r"C:\Git_Dan\arraytools\Data\numpy_demos.gdb\sample_10k"
#    in_flds = ['OBJECTID', 'County', 'Town']
#    a = arcpy.da.TableToNumPyArray(in_tbl, in_flds)
