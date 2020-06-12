# -*- coding: UTF-8 -*-
"""
:Script:   arcpytools_plt.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-09-12
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import os
from textwrap import dedent
import warnings

import numpy as np
import arcpy
# from arcpytools import array_fc, array_struct, tweet

warnings.simplefilter('ignore', FutureWarning)

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['_col_format', '_describe', '_xyID',
           'arr2line', 'arr2pnts', 'arr2polys',
           'array_fc', 'array_struct', 'fc_array',
           'fc_info', 'frmt_rec',
           'output_polygons', 'output_polylines',
           'pd_', 'shapes2fc', 'tweet']


def tweet(msg):
    """Produce a message for both arcpy and python
    : msg - a text message
    """
    m = "{}".format(msg)
    arcpy.AddMessage(m)
    print(m)


def _describe(in_fc=None):
    """Simply return the arcpy.da.Describe object
    : desc.keys() an abbreviated list...
    : [... 'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    :  ... 'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    :  'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    :  'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    :  'spatialReference',  ...]
    """
    if in_fc is not None:
        in_fc = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
        return arcpy.da.Describe(in_fc)
    else:
        return None

def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including...

    Parameters:
    -----------
    - shp_fld  :
        field name which contains the geometry object
    - oid_fld  :
        the object index/id field name
    - SR       :
        spatial reference object (use SR.name to get the name)
    - shp_type :
        shape type (Point, Polyline, Polygon, Multipoint, Multipatch)
    - others   :
        'areaFieldName', 'baseName', 'catalogPath','featureType','fields',
        'hasOID', 'hasM', 'hasZ', 'path'

     - all_flds :
         [i.name for i in desc['fields']]
    """
    desc = _describe(in_fc)
    args = ['shapeFieldName', 'OIDFieldName', 'shapeType', 'spatialReference']
    shp_fld, oid_fld, shp_type, SR = [desc[i] for i in args]
    if prn:
        frmt = "FeatureClass:\n   {}".format(in_fc)
        f = "\n{!s:<16}{!s:<14}{!s:<10}{!s:<10}"
        frmt += f.format(*args)
        frmt += f.format(shp_fld, oid_fld, shp_type, SR.name)
        tweet(frmt)
    else:
        return shp_fld, oid_fld, shp_type, SR


# ---- geometry related -----------------------------------------------------
#
def get_polys(in_fc):
    """Return polygons from a polygon featureclass
    """
    out_polys = []
    out_ids = []
    with arcpy.da.SearchCursor(in_fc,  ["SHAPE@", "OID@"]) as cursor:
        for row in cursor:
            out_polys.append(row[0])
            out_ids.append(row[1])
    return out_polys, out_ids


def _poly_ext(p):
    """poly* extent
    """
    L, B = p.extent.lowerLeft.X, p.extent.lowerLeft.Y
    R, T = p.extent.upperRight.X, p.extent.upperRight.Y
    return L, B, R, T


def trans_rot(a, angle):
    """simplified translate and rotate
    """
    cent = a.mean(axis=0)
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s,  c)))
    return  np.einsum('ij,kj->ik', a - cent, R) + cent


def cal_area(poly, cuts, cutters, factor):
    """Calculate the areas
    """
    tot_area = poly.area
    fract_areas = np.array([c.area/tot_area for c in cuts])
    c_sum = np.cumsum(fract_areas)
    f_list = np.linspace(0.0, 1.0, factor, endpoint=False)
    idxs = [np.argwhere(c_sum <= i)[-1][0] for i in f_list[1:]]
    out = []
    for idx in idxs:
        c_line = cutters[idx]
        left, right = poly.cut(c_line)
        out.append([left.area, c_line])
    return out


def _xyID(in_fc, to_pnts=True):
    """Convert featureclass geometry (in_fc) to a simple 2D structured array
    :  with ID, X, Y values. Optionally convert to points, otherwise centroid.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    a.dtype = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    del cur
    return a


def array_struct(a, fld_names=['X', 'Y'], dt=['<f8', '<f8']):
    """Convert an array to a structured array

    Parameters:
    -----------
    - a : an ndarray with shape at least (N, 2)
    -  dt : dtype class
    -  names : names for the fields
    """
    dts = [(fld_names[i], dt[i]) for i in range(len(fld_names))]
    z = np.zeros((a.shape[0],), dtype=dts)
    names = z.dtype.names
    for i in range(a.shape[1]):
        z[names[i]] = a[:, i]
    return z


def array_fc(a, out_fc, fld_names, SR):
    """Array to featureclass/shapefile...optionally including all fields

    Parameters:
    -----------
    - out_fc :  featureclass/shapefile... complete path
    - fld_names : the Shapefield name ie ['Shape'] or ['X', 'Y's]
    - SR : spatial reference of the output

    See also :
        NumpyArrayToFeatureClass, ListFields for information and options
    """
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.da.NumPyArrayToFeatureClass(a, out_fc, fld_names, SR)
    return out_fc


def fc_array(in_fc, flds="*", allpnts=True):
    """Convert a featureclass to an ndarray...with optional fields besides the
    FID/OIDName and Shape fields.

    Parameters:
    -----------
    in_fc : text
        Full path to the geodatabase and the featureclass name

    flds : text or list
        - ``''   : just an object id and shape field``
        - ``'*'  : all fields in the featureclass or``
        - ``list : specific fields ['OBJECTID','Shape','SomeClass', etc]``

    allpnts : boolean
        - True `explodes` geometry to individual points.
        - False returns the centroid

    Requires:
    ---------
        fc_info(in_fc) function

    See also:
    ---------
        FeatureClassToNumPyArray, ListFields for more information in current
        arcpy documentation
    """
    out_flds = []
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)  # get the base information
    flds_all = arcpy.ListFields(in_fc)
    flds_oth = [f for f in flds_all if f.type not in ('OID', 'Geometry')]
    fld_names = [f.name for f in flds_oth
                 if f.name not in ["Shape_Length", "Shape_Area", "Shape"]]
    oid_geom = [oid_fld, 'SHAPE@X', 'SHAPE@Y']
    nulls = {'Double':np.nan,
             'Single':np.nan,
             'Short':np.iinfo(np.int32).min,
             'Long':np.iinfo(np.int32).min,
             'Float':np.nan,
             'Integer':np.iinfo(np.int32).min,
             'String':str(None),
             'Text':str(None)}
    fld_dict = {i.name: i.type for i in flds_oth}
    null_dict = {f:nulls[fld_dict[f]] for f in fld_names}
    if flds == "":                        # return just OID and Shape values
        out_flds = oid_geom  # FID and Shape X, Y
    elif flds == "*":                     # all fields
        out_flds = oid_geom + fld_names
    else:
        out_flds = [oid_fld, 'SHAPE@X', 'SHAPE@Y']
        for f in flds_oth:
            if f.name in flds:
                out_flds.append(f.name)
#    pth = os.path.dirname(in_fc)
#    out_flds = [arcpy.ValidateFieldName(f, pth) for f in out_flds]
    frmt = """\nRunning 'fc_array' with ....
    \nfeatureclass... {}\nFields... {}\nAll pnts... {}\nSR... {}
    """
    args = [in_fc, out_flds, allpnts, SR.name]
    msg = dedent(frmt).format(*args)
    tweet(msg)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                          field_names=out_flds,
                                          where_clause="",
                                          spatial_reference=SR,
                                          explode_to_points=allpnts,
                                          skip_nulls=False,
                                          null_value=null_dict)
    # out it goes in array format
    return a, out_flds, SR


def arr2pnts(in_fc, as_struct=True, shp_fld=None, SR=None):
    """Create points from an array.
    :  in_fc - input featureclass
    :  as_struct - if True, returns a structured array with X, Y fields,
    :            - if False, returns an ndarray with dtype='<f8'
    :Notes: calls fc_info to return featureclass information
    """
    if shp_fld is None or SR is None:
        shp_fld, oid_fld, SR = fc_info(in_fc)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR)
    dt = [('X', '<f8'), ('Y', '<f8')]
    if as_struct:
        shps = np.array([tuple(i) for i in a[shp_fld]], dtype=dt)
    else:
        shps = a[shp_fld]
    return shps, shp_fld, SR


def arr2line(a, out_fc, SR=None):
    """create lines from an array"""
    pass


def shapes2fc(shps, out_fc):
    """Create a featureclass/shapefile from poly* shapes.
    :  out_fc - full path and name to the output container (gdb or folder)
    """
    msg = "\nCan't overwrite the {}... rename".format(out_fc)
    try:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(shps, out_fc)
    except ValueError:
        tweet(msg)


def arr2polys(a, out_fc, oid_fld, SR):
    """Make poly* features from a structured array.
    :  a - structured array
    :  out_fc: a featureclass path and name, or None
    :  oid_fld - object id field, used to partition the shapes into groups
    :  SR - spatial reference object, or name
    :Returns:
    :-------
    :  Produces the featureclass optionally, but returns the polygons anyway.
    """
    arcpy.overwriteOutput = True
    pts = []
    keys = np.unique(a[oid_fld])
    for k in keys:
        w = np.where(a[oid_fld] == k)[0]
        v = a['Shape'][w[0]:w[-1] + 1]
        pts.append(v)
    # Create a Polygon from an Array of Points, save to featueclass if needed
    s = []
    for pt in pts:
        s.append(arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pt]), SR))
    return s


def output_polylines(out_fc, SR, pnt_groups):
    """Produce the output polygon featureclass.
    :Requires:
    :--------
    : - A list of lists of points
    :   aline = [[[0, 0], [1, 1]]]  # a list of points
    :   aPolyline = [[aline]]       # a list of lists of points
    """
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg
    polylines = []
    for pnts in pnt_groups:
        for pair in pnts:
            arr = arcpy.Array([arcpy.Point(*xy) for xy in pair])
            pl = arcpy.Polyline(arr, SR)
            polylines.append(pl)
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(polylines, out_fc)
    return


def output_polygons(out_fc, SR, pnt_groups):
    """Produce the output polygon featureclass.

    Parameters:
    -----------
    out_fc : string
        The path and name of the featureclass to be created.
    SR : spatial reference of the output featureclass
    pnts_groups :
        The point groups, list of lists of points, to include parts rings.

    Requires:
    --------

    - A list of lists of points.  Four points form a triangle is the minimum
    -  aline = [[0, 0], [1, 1]]  # a list of points
    -  aPolygon = [aline]        # a list of lists of points
    """
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg
    polygons = []
    for pnts in pnt_groups:
        for pair in pnts:
            arr = arcpy.Array([arcpy.Point(*xy) for xy in pair])
            pl = arcpy.Polygon(arr, SR)
            polygons.append(pl)
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(polygons, out_fc)
    return

# ---- formatting, from arraytools ------------------------------------------
#
# ----------------------------------------------------------------------
# (4) frmt_rec .... code section
#  frmt_rec requires _col_format
def _col_format(a, c_name="c00", deci=0):
    """Determine column format given a desired number of decimal places.
    Used by frmt_struct.

    `a` : column
        A column in an array.
    `c_name` : text
        column name
    `deci` : int
        Desired number of decimal points if the data are numeric

    Notes:
    -----
        The field is examined to determine whether it is a simple integer, a
        float type or a list, array or string.  The maximum width is determined
        based on this type.

        Checks were also added for (N,) shaped structured arrays being
        reformatted to (N, 1) shape which sometimes occurs to facilitate array
        viewing.  A kludge at best, but it works for now.
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
    # ---- lists, arrays, strings. Check for (N,) vs (N,1)
    # I made some changes in how col_wdth is determined, old is commented
    else:
        if a.ndim == 1:  # ---- check for (N, 1) format of structured array
            a = a[0]
        dt = a.dtype.descr[0][1]
        col_wdth = int("".join([i for i in dt if i.isdigit()]))
#       col_wdth = max([len(str(i)) for i in a])
        col_wdth = max(len(c_name), col_wdth) + 1  # + deci
        c_fmt = "!s:>" + "{}".format(col_wdth)
    return c_fmt, col_wdth


def pd_(a, deci=2, use_names=True, prn=True):
    """see help for `frmt_rec`..."""
    ret = frmt_rec(a, deci=deci, use_names=use_names, prn=prn)
    return ret


def frmt_rec(a, deci=2, use_names=True, prn=True):
    """Format a structured array with a mixed dtype.

    NOTE : Can be called as `pd_(a, ... )` to emulate pandas dataframes
        You should limit large arrays to a slice ie. a[:50]

    Requires:
    -------
    `a` : array
        A structured/recarray
    `deci` : int
        To facilitate printing, this value is the number of decimal
        points to use for all floating point fields.
    `use_names` : boolean
        If no names are available, then create them
    `prn` : boolean
        True to print, False to return the string
    Notes:
    -----
        `_col_format` : does the actual work of obtaining a representation of
        the column format.

        It is not really possible to deconstruct the exact number of decimals
        to use for float values, so a decision had to be made to simplify.
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

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
#    gdb_fc = ['Data', 'point_tools.gdb', 'radial_pnts']
#    in_fc = "/".join(script.split("/")[:-2] + gdb_fc)
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
#    result = fc_array(in_fc, flds="", allpnts=True)  # a, out_flds, SR
