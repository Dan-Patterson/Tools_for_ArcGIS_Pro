# -*- coding: UTF-8 -*-ct
"""
:Script:   rasterstats.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-13
:Purpose:  tools for working with numpy arrays
:Useage:
:
:Requires:
:  arraytools.tools - nd2struct, stride
:References:
: https://community.esri.com/blogs/dan_patterson/2018/02/06/
:       cell-statistics-made-easy-raster-data-over-time
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent, indent
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=100, precision=2, suppress=True,
                    threshold=150, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['check_shapes', 'check_stack', 'mask_stack',
           'stack_sum', 'stack_cumsum',  # statistical functions
           'stack_prod', 'stack_cumprod',
           'stack_min', 'stack_mean',
           'stack_median', 'stack_max',
           'stack_std', 'stack_var',
           'stack_percentile',
           'stack_stats',
           'stack_stats_tbl']


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


# ----------------------------------------------------------------------
# (4) frmt_rec .... code section
#  frmt_rec requires _col_format
def _col_format(a, c_name="c00", deci=0):
    """Determine column format given a desired number of decimal places.
    :  Used by frmt_struct.
    :  a - a column in an array
    :  c_name - column name
    :  deci - desired number of decimal points if the data are numeric
    :Notes:
    :-----
    :  The field is examined to determine whether it is a simple integer, a
    :  float type or a list, array or string.  The maximum width is determined
    :  based on this type.
    :  Checks were also added for (N,) shaped structured arrays being
    :  reformatted to (N, 1) shape which sometimes occurs to facilitate array
    :  viewing.  A kludge at best, but it works for now.
    """
    a_kind = a.dtype.kind
    if a_kind in ('i', 'u'):  # ---- integer type
        w_, m_ = [':> {}.0f', '{:> 0.0f}']
        col_wdth = len(m_.format(a.max())) + 1
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = w_.format(col_wdth, 0)
    elif a_kind == 'f' and np.isscalar(a[0]):  # ---- float type with rounding
        w_, m_ = [':> {}.{}f', '{:> 0.{}f}']
        a_max, a_min = np.round(np.sort(a[[0, -1]]), deci)
        col_wdth = max(len(m_.format(a_max, deci)),
                       len(m_.format(a_min, deci))) + 1
        col_wdth = max(len(c_name), col_wdth) + 1
        c_fmt = w_.format(col_wdth, deci)
    else:  # ---- lists, arrays, strings. Check for (N,) vs (N,1)
        if a.ndim == 1:  # ---- check for (N, 1) format of structured array
            a = a[0]
        col_wdth = max([len(str(i)) for i in a])
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = "!s:>" + "{}".format(col_wdth)
    return c_fmt, col_wdth


def frmt_rec(a, deci=2, use_names=True, prn=True):
    """Format a structured array with a mixed dtype.
    :Requires
    :-------
    : a - a structured/recarray
    : deci - to facilitate printing, this value is the number of decimal
    :        points to use for all floating point fields.
    : _col_format - does the actual work of obtaining a representation of
    :  the column format.
    :Notes
    :-----
    :  It is not really possible to deconstruct the exact number of decimals
    :  to use for float values, so a decision had to be made to simplify.
    """
    dt_names = a.dtype.names
    N = len(dt_names)
    c_names = [["C{:02.0f}".format(i) for i in range(N)], dt_names][use_names]
    # ---- get the column formats from ... _col_format ----
    dts = []
    wdths = []
    pair = list(zip(dt_names, c_names))
    for i in range(len(pair)):
        fld, nme = pair[i]
        c_fmt, col_wdth = _col_format(a[fld], c_name=nme, deci=deci)
        dts.append(c_fmt)
        wdths.append(col_wdth)
    row_frmt = " ".join([('{' + i + '}') for i in dts])
    hdr = ["!s:>" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = " ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = "--n--" + hdr2.format(*c_names)
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    # ---- check for structured arrays reshaped to (N, 1) instead of (N,) ----
    len_shp = len(a.shape)
    idx = 0
    for i in range(a.shape[0]):
        if len_shp == 1:  # ---- conventional (N,) shaped array
            row = " {:03.0f} ".format(idx) + row_frmt.format(*a[i])
        else:             # ---- reformatted to (N, 1)
            row = " {:03.0f} ".format(idx) + row_frmt.format(*a[i][0])
        idx += 1
        txt.append(row)
    msg = "\n".join([i for i in txt])
    if prn:
        print(msg)
    else:
        return msg


# ---- raster checks --------------------------------------------------------
#
def rasterfile_info(fname, prn=False):
    """Obtain raster stack information from the filename of an image
    :
    """
    #
    frmt = """\nFile path - {}\nName - {}\nSpatial Ref - {}\nRaster type - {}
NoData - {}\nIs integer? - {}\nBands - {}\nCell h {} w {}
Lower Left X {} Y {}\nExtent  h {} w {}"""
    desc = arcpy.Describe(fname)
    r_data_type = desc.datasetType  # 'RasterDataset'
    args = []
    if r_data_type == 'RasterDataset':
        r = arcpy.Raster(fname)
        r.catalogPath            # full path name and file name
        pth = r.path             # path only
        name = r.name            # file name
        SR = r.spatialReference
        r_type = r.format        # 'TIFF'
        #
        nodata = r.noDataValue
        is_int = r.isInteger
        bands = r.bandCount
        cell_hght = r.meanCellHeight
        cell_wdth = r.meanCellWidth
        extent = desc.Extent
        LL = extent.lowerLeft  # Point (X, Y, #, #)
        hght = r.height
        wdth = r.width
        args = [pth, name, SR.name, r_type, nodata, is_int, bands,
                cell_hght, cell_wdth, LL.X, LL.Y, hght, wdth]
    if prn:
        tweet(dedent(frmt).format(*args))
    else:
        return args


# ---- array checks and creation --------------------------------------------
# ---- 3D arrays for stacked operations
#
def check_shapes(arrs):
    """Check the shapes of the arrays to ensure they are all equal
    """
    shps = [i.shape for i in arrs]
    eq = np.all(np.array([shps[0] == i for i in shps[1:]]))
    err = "Arrays arr not of the same shape..."
    if not eq:
        raise ValueError("{}\n{}".format(err, shps))


def check_stack(arrs):
    """Do the basic checking of the stack to ensure that a 3D array is
    :  generated
    """
    err1 = "Object, structured arrays not supported, current type..."
    err2 = "3D arrays supported current ndim..."
    if isinstance(arrs, (list, tuple)):
        arrs = np.array(arrs)
    if arrs.dtype.kind in ('O', 'V'):
        raise ValueError("{} {}".format(err1, arrs.dtype.kind))
    if arrs.ndim != 3:
        raise ValueError("{} {}".format(err2, arrs.ndim))
    return arrs


def mask_stack(arrs, nodata=None):
    """Produce masks for a 3d array"""
    if nodata is None:
        return arrs
    m = (arrs[:, ...] == nodata).any(0)
    msk = [m for i in range(arrs.shape[0])]
    msk = np.array(msk)
    a_m = np.ma.MaskedArray(arrs, mask=msk)
    return a_m


# ---- Statistics for stacked arrays (3D) ------------------------------------
#
def stack_sum(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nansum(a, axis=0)


def stack_cumsum(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nancumsum(a, axis=0)


def stack_prod(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanprod(a, axis=0)


def stack_cumprod(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nancumprod(a, axis=0)


def stack_min(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmin(a, axis=0)


def stack_mean(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmean(a, axis=0)


def stack_median(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmedian(a, axis=0)


def stack_max(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmax(a, axis=0)


def stack_std(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanstd(a, axis=0)


def stack_var(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanvar(a, axis=0)


def stack_percentile(arrs, q=50, nodata=None):
    """nanpercentile for an array stack with optional nodata masked
    :  arrs - either a list, tuple of arrays or an array with ndim=3
    :  q - the percentile
    :  nodata - nodata value, numeric or np.nan (will upscale integers)
    """
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    nan_per = np.nanpercentile(a, q=q, axis=0)
    return nan_per


def stack_stats(arrs, ax=0, nodata=None):
    """All statistics for arrs
    :
    :  arrs - either a list, tuple of arrays or an array with ndim=3
    :  ax - axis, either, 0 (by band) or (1,2) to get a single value for
    :       each band
    :  nodata - nodata value, numeric or np.nan (will upscale integers)
    """
    arrs = check_stack(arrs)
    a_m = mask_stack(arrs, nodata=nodata)
    nan_sum = np.nansum(a_m, axis=ax)
    nan_min = np.nanmin(a_m, axis=ax)
    nan_mean = np.nanmean(a_m, axis=ax)
    nan_median = np.nanmean(a_m, axis=ax)
    nan_max = np.nanmax(a_m, axis=ax)
    nan_std = np.nanstd(a_m, axis=ax)
    nan_var = np.nanvar(a_m, axis=ax)
    stats = [nan_sum, nan_min, nan_mean, nan_median, nan_max, nan_std, nan_var]
    if len(ax) == 1:
        nan_cumsum = np.nancumsum(a_m, axis=ax)
        stats.append(nan_cumsum)
    return stats


def stack_stats_tbl(arrs, nodata=None):  # col_names, args):
    """Produce the output table
    :   ('N_', '<i4'), ('N_nan', '<i4')
    """
    stats = stack_stats(arrs, ax=(1, 2), nodata=nodata)
    d = [(i, '<f8')
         for i in ['Sum', 'Min', 'Mean', 'Med', 'Max', 'Std', 'Var']]
    dts = [('Band', '<i4'), ('N', '<i4'), ('N_nan', '<i4')] + d
    N, r, c = arrs.shape
    cols = len(dts)
    z = np.empty(shape=(N,), dtype=dts)
    z[z.dtype.names[0]] = np.arange(0, N)
    z[z.dtype.names[1]] = np.array([r*c]*N)
    z[z.dtype.names[2]] = np.count_nonzero(arrs == nodata, axis=(1, 2))
    for i in range(cols-3):
        z[z.dtype.names[i+3]] = stats[i]
    return z


def _tool():
    """run when script is from a tool
    """
    in_rast = sys.argv[1]
    stat = sys.argv[2]
    out_tbl = sys.argv[3]
    deci = sys.argv[4]
    tweet("in_rast {}\nstats: {}\nout_tbl: {}".format(in_rast, stat, out_tbl))
    #
    if isinstance(in_rast, (list, tuple)):
        if len(in_rast) == 1:
            in_rast = [in_rast]
            arrs = tifffile_arr(in_rast, bandXY=False)
    else:
        arrs = tifffile_arr(in_rast,  bandXY=False)
        if arrs.ndim < 3:
            arrs = arrs.reshape(1, arrs.shape[0], arrs.shape[1])
    z = stack_stats_tbl(arrs)
    if not (out_tbl in ("#", None)):
        arcpy.NumPyArrayToTable(z, out_tbl)
    if isinstance(deci, (str, np.str)):
        if deci.isdigit():
            deci = int(deci)
    if isinstance(deci, (int, float)):
        deci = int(deci)
    else:
        deci = 2
    msg = frmt_rec(z, deci=deci, use_names=True, prn=False)
    tweet(msg)
#    no_data = sys.argv[5]
#    out_rast = sys.argv[6]
#    SR = sys.argv[7]


def _demo_stack():
    """demo stack
    :
    """
#    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\Array_small_31.npy"
    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\Array_10x10_31.npy"
    stack = np.load(fname)
#    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\Array_small_31.tif"
#    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\Arr_10_10_composite.tif"
#    stack = arcpy.RasterToNumPyArray(fname)
#    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\Arr_10_10_composite.tif"
    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\Array_100x100_31.tif"
    import warnings
    warnings.filterwarnings('ignore')
    from tifffile import TiffFile
    with TiffFile(fname) as tif:
        img = tif.asarray()
    #img = np.rollaxis(img, axis=2, start=0)
    img = np.ascontiguousarray(img, img.dtype)
    return img


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


def tifffile_arr(fname, bandXY=True):
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
files = "c:/temp/stats.tif"
if not hasattr(files, 'seek') and len(files) == 1:
    files = files[0]
if isinstance(files, basestring) or hasattr(files, 'seek'):
    with TiffFile(files, **kwargs_file) as tif:
        return tif.asarray(**kwargs)
    """
    from tifffile import imread, TiffFile
    a = imread(fname)
    if not bandXY:
        with TiffFile(fname) as tif:
            a = tif.asarray()
            a = np.rollaxis(a, axis=2, start=0)
    return a


# ----------------------------------------------------------------------
# .... final code section
if len(sys.argv) == 1:
    testing = True
#    a = _demo_stack()
    # fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\a30_compbnds.tif"
    #fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\a30_tifffile.tif"
    fname = r"C:\GIS\Tools_scripts\Raster_tools\Data\rast_composite_9bands.tif"
    a = tifffile_arr(fname, bandXY=True)
    frmt = "Result...\n{}"
#    print(frmt.format(a))
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
#    stack = _demo_stack()
