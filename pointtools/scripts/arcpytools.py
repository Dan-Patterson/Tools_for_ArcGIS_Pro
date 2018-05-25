# -*- coding: UTF-8 -*-
"""
arcpytools
==========

Script :   arcpytools.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-03-31

Purpose :  tools for working with numpy arrays and arcpy

Useage :

References :

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
import arcpy
# from arcpytools import array_fc, array_struct, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['_describe',
           '_xyID',
           'arr2line',
           'arr2pnts',
           'arr2polys',
           'array_fc',
           'array_struct',
           'fc_array',
           'fc_info',
           'output_polygons',
           'output_polylines',
           'shapes2fc', 'tweet'
           ]


def tweet(msg):
    """Produce a message for both arcpy and python
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
#    return None


def _describe(in_fc):
    """Simply return the arcpy.da.Describe object
    : desc.keys() an abbreviated list...
    : [... 'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    :  ... 'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    :  'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    :  'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    :  'spatialReference',  ...]
    """
    return arcpy.da.Describe(in_fc)


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


def _xyID(in_fc, to_pnts=True):
    """Convert featureclass geometry (in_fc) to a simple 2D structured array
    with ID, X, Y values. Optionally convert to points, otherwise centroid.
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


def fc_array(in_fc, flds, allpnts):
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
    fields = arcpy.ListFields(in_fc)      # all fields in the shapefile
    if flds == "":                        # return just OID and Shape field
        out_flds = [oid_fld, shp_fld]     # FID and Shape field required
    elif flds == "*":                     # all fields
        out_flds = [f.name for f in fields]
    else:
        out_flds = [oid_fld, shp_fld]
        for f in fields:
            if f.name in flds:
                out_flds.append(f.name)
    frmt = """\nRunning 'fc_array' with ....
    \nfeatureclass... {}\nFields... {}\nAll pnts... {}\nSR... {}
    """
    args = [in_fc, out_flds, allpnts, SR.name]
    msg = dedent(frmt).format(*args)
    tweet(msg)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, out_flds, "", SR, allpnts)
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
    -  aPolyline = [aline]        # a list of lists of points
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


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
    gdb_fc = ['Data', 'point_tools.gdb', 'radial_pnts']
    in_fc = "/".join(script.split("/")[:-2] + gdb_fc)
    result = fc_array(in_fc, flds="", allpnts=True)  # a, out_flds, SR
