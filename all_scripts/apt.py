# -*- coding: UTF-8 -*-
"""
apt.py ... arcpy tools
======================

Script :   apt.py ... (arcpytools.py)

Author :   Dan.Patterson@carleton.ca

Modified : 2018-09-12

Purpose :  tools for working with arcpy and numpy arrays

Notes
-----
Functions
---------

-  _arr_common : common function for poly* features
-  _shapes_fc : convert shapes to featureclass
-  arr_pnts : array to points
-  arr_polyline : array to polyline
-  arr_polygon : array to polygon
-  array_fc : array to featureclass
-  array_struct : array to structured array
-  pnts_arr : points to array
-  polylines_arr : polylines to array
-  polygons_arr : polygons to array
-  fc_array : featureclass to array
-  change_fld : convert arc to np field types
-  tbl_arr : convert a table to an array
-  tbl_2_np_array : shortcut to TableToNumPyArray
-  to_fc : convert the results back to a featureclass

**Common variables used in the functions**

1. Array variables/properties
  - a  : structured/recarray
  - dt : array dtype properties

2. Featureclass variables/properties

  -  shps    - point geometry objects needed to create the featureclass
  -  in_fc   - input featureclass
  -  out_fc  - full path and name to the output container (gdb or folder)
  -  SR      - spatial reference object (use SR.name to get the name)
  -  shp_fld - field name which contains the geometry object
  -  oid_fld - the object index/id field name

3. Field options:

    ['OBJECTID', 'Shape'], ['OID@', 'SHAPE@XY'],
    ['OID@', 'SHAPE@X', 'SHAPE@Y'], ['OID@', 'SHAPE@JSON']

  >>> cur = arcpy.da.SearchCursor(in_fc, field_names, .....)
  >>> cur = arcpy.da.SearchCursor(in_fc, ['OBJECTID', 'Shape'], None, None,
                                  True, (None, None))

Procedures:
-----------

1. Split featureclass to array

  - Split the geometry
  - Split the attributes (TableToNumPyArray)

2. Create featureclass

  - geometry : arr_pnts, arr_polylines, arr_polygons
  - join attributes : ExtendTable

References:
-----------

(1) main link section:  http://pro.arcgis.com/en/pro-app/arcpy/data-access/

(2) ...extendtable.htm

>>> ExtendTable (in_table, table_match_field, in_array,
                 array_match_field, {append_only})

(3) ...featureclasstonumpyarray.htm

>>> FeatureClassToNumPyArray(in_table, field_names, {where_clause},
                            {spatial_reference}, {explode_to_points},
                            {skip_nulls}, {null_value})

(4) ...numpyarraytofeatureclass.htm

>>>  NumPyArrayToFeatureClass(in_array, out_table, shape_fields,
                              {spatial_reference})

(5) ...tabletonumpyarray.htm

>>> TableToNumPyArray(in_table, field_names, {where_clause},
                     {skip_nulls}, {null_value})

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
from arraytools.fc_tools._common import fc_info, tweet
import arcpy


__all__ = ['_arr_common', '_split_array', 'shapes_fc',
           'arr_pnts', 'obj_polyline', 'obj_polyline',
           'struct_polyline', 'struct_polygon',
           'array_fc', 'array_struct', 'arc_np',
           'pnts_arr', 'arr_polyline_fc', 'arr_polygon_fc',
           'shapes_fc', '_id_geom_array',
           'polylines_arr', 'polygons_arr',
           'fc_array', 'change_fld',
           'tbl_arr', 'to_fc'
           ]

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=3, suppress=True,
                    threshold=20, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- Array to featureclass section ----

def _arr_common(a, oid_fld, shp_fld):
    """Common structure for polyline, polygon and multipoint features.
    This functions pulls out all the array points that belong to a
    feature so they can be used in shape construction.

    Requires:
    --------
    a : structured/recarray

    oid_fld : the object index/id field name
        Normally the `OBJECTID` field, but any field that provides a means to
        determine sequential objects.

    shp_fld : the table field which contains the geometry
        Usually the `Shape` field.
    """
    pts = []
    keys = np.unique(a[oid_fld])
    for k in keys:
        w = np.where(a[oid_fld] == k)[0]
        v = a[shp_fld][w[0]:w[-1] + 1]
        pts.append(v)
    return pts


# @time_deco
def _split_array(a, fld='ID'):
    """Split a structured/recarray array, using unique values in a numeric id
    `fld` assumed to be sorted in the correct order which indicates the
    group a record belongs to.  From 'arraytools split_array'.
    """
    return np.split(a, np.where(np.diff(a[fld]))[0] + 1)


def arr_pnts(a, out_fc, shp_fld, SR):
    """Make point features from a structured array.

    Requires:
    --------
    `a` : structured/recarray

    `out_fc` : featureclass full path and name

    `shp_fld` : the table field which contains the geometry

    `SR` : the spatial reference/coordinate system of the geometry
    """
    msg0 = "\nCreated..\n{}".format(out_fc)
    msg1 = "\nCan't overwrite {}... rename it".format(out_fc)
    try:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.da.NumPyArrayToFeatureClass(a, out_fc, shp_fld, SR)
        tweet(msg0)
    except Exception as e:
        tweet(msg1 + str(e))


def obj_polyline(pnts, SR=None):
    """Object array of point geometry X, Y coordinates to a polyline, using
    a known spatial reference (SR)
    """
    f = []
    for pt in pnts:
        f.append(arcpy.Polyline(arcpy.Array([arcpy.Point(*p)
                                            for p in pt.tolist()]), SR))
    return f


def obj_polygon(pnts, SR=None):
    """Object array of point geometry X, Y coordinates to a polygon, using
    a known spatial reference (SR)
    """
    f = []
    for pt in pnts:
        f.append(arcpy.Polygon(arcpy.Array([arcpy.Point(*p)
                                            for p in pt.tolist()]), SR))
    return f


def struct_polyline(a, oid_fld, shp_fld, SR):
    """Make polyline features from a structured array.

    Requires:
    --------
    `a`       : structured array

    `out_fc`  : a featureclass path and name, or None

    `oid_fld` : object id field, used to partition the shapes into groups

    `shp_fld` : shape field(s)

    `SR`      : spatial reference object, or name

    Returns:
    -------
        Produces the featureclass optionally, but returns the polygons anyway.
    """
    pts = _arr_common(a, oid_fld, shp_fld)
    f = []
    for pt in pts:
        f.append(arcpy.Polyline(arcpy.Array([arcpy.Point(*p)
                                            for p in pt.tolist()]), SR))
    return f


def struct_polygon(a, out_fc, oid_fld, shp_fld, SR):
    """Make polygon features from a structured array.

    Requires:
    --------
    `a` : structured array

    `out_fc` : a featureclass path and name, or None

    `oid_fld` : object id field, used to partition the shapes into groups

    `shp_fld` : shape field(s)

    `SR` : spatial reference object, or name

    Returns:
    -------
        Produces the featureclass optionally, but returns the polygons anyway.
    """
    pts = _arr_common(a, oid_fld, shp_fld)
    f = []
    for pt in pts:
        f.append(arcpy.Polygon(arcpy.Array([arcpy.Point(*p)
                                            for p in pt.tolist()]), SR))
    return f


def array_fc(a, out_fc=None, shp_fld=['Shape'], SR=None):
    """Array to featureclass/shapefile...optionally including all fields

    Requires:
    --------

    a : structured or recarray

    out_fc : the full path and filename to a featureclass or shapefile

    shp_fld : shapefield name(s) ie. ['Shape'] or ['X', 'Y']

    References:
    ----------
        NumpyArrayToFeatureClass, ListFields for information and options

    """
    if not out_fc:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.da.NumPyArrayToFeatureClass(a, out_fc, shp_fld, SR)
    # return out_fc


def array_struct(a, fld_names=['X', 'Y'], dt=['<f8', '<f8']):
    """Convert an array to a structured array

    Requires:
    --------
    `a` : an ndarray with shape at least (N, 2)

    `dt` : dtype class

    `names` : names for the fields
    """
    dts = [(fld_names[i], dt[i]) for i in range(len(fld_names))]
    z = np.zeros((a.shape[0],), dtype=dts)
    names = z.dtype.names
    for i in range(a.shape[1]):
        z[names[i]] = a[:, i]
    return z


# ---- Featureclass to array section ----

def arc_np(in_fc):
    """Alternate constructor to create a structured array from a feature class.
    This function only returns the geometry portion.  The OID field is
    retained should you wish to associate attributes back to the results.
    There is no point in carrying extra baggage around when you don't need it.

    Requires:
    --------
    `in_fc` : the file path to the feature class

    ArcGIS Pro is assumed, since the new arcpy.da.Describe is used which
    returns a dictionary of properties.

    Returns:
    -------
    A structured array with a specified dtype to facilitate geometry
    reconstruction later.  The attributes are kept separate.

    Alternatives:
    ------------
    SpatialDataFrame from ArcGIS API for Python... or GeoPandas
    """
    desc = arcpy.da.Describe(in_fc)
    shp_type = desc['shapeType']
    prefix = desc['shapeFieldName']
    fields = ['OID@', prefix + '@']
    dt = [('ID_num', '<i4'), ('Part_num', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    pnts = []
    with arcpy.da.SearchCursor(in_fc, fields) as cursor:
        for row in cursor:
            oid, shp = row
            for j in range(len(shp)):
                pt = shp.getPart(j)
                if shp_type in ('Point', 'point'):
                    pnts.extend([(oid, j, pt.X, pt.Y)])
                else:
                    p_list = [(oid, j, pnt.X, pnt.Y) for pnt in pt if pnt]
                    pnts.extend(p_list)
    a = np.asarray(pnts, dtype=dt)
    return a


def pnts_arr(in_fc, as_struct=True, shp_fld=None, SR=None):
    """Create points from an array.

    Requires:
    --------

    `in_fc` :
        input featureclass

    `as_struct` : boolean
        - True, returns a structured array with Id, X, Y fields
        - False, returns an ndarray with dtype='<f8'

    `shp_fld` & `SR` :
        if unspecified, it will be determined using fc_info
        to return featureclass information.
    """
    if shp_fld is None or SR is None:
        shp_fld, oid_fld, SR, shp_type = fc_info(in_fc)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR)
    dt = [('Id', '<i4'), ('X', '<f8'), ('Y', '<f8')]
    if as_struct:
        shps = np.array([i for i in a[[oid_fld, shp_fld]]], dtype=dt)
    else:
        shps = a[shp_fld]
    return shps


# ---- points to point, polyline or polygon featureclass----------------------
#
def output_points(out_fc, SR, pnts):
    """Produce the output point featureclass"""
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg
    pnts_lst = []
    for pnt in pnts:                 # create the point geometry
        pnts_lst.append(arcpy.PointGeometry(arcpy.Point(*pnt), SR))
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(pnts_lst, out_fc)
    return out_fc


def output_polylines(out_fc, SR, pnts):
    """Produce the output polygon featureclass.
    :Requires:
    :--------
    : - A list of lists of points
    :   aline = [[0, 0], [1, 1]]  # a list of points
    :   aPolyline = [aline]       # a list of lists of points
    """
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg
    polylines = []
    for pair in pnts:
        pl = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in pair]), SR)
        polylines.append(pl)
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(polylines, out_fc)
    return


def output_polygons(out_fc, SR, pnts):
    """Produce the output polygon featureclass.

    Requires:
    --------
        - A list of lists of points
        - aline = [[0, 0], [1, 1]]  # a list of points
        - aPolygon = [aline]       # a list of lists of points
    """
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg
    polygons = []
    for pair in pnts:
        pl = arcpy.Polygon(arcpy.Array([arcpy.Point(*xy) for xy in pair]), SR)
        polygons.append(pl)
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(polygons, out_fc)
    return


# ---- piece the geometry to featureclasses --------------------------------
def arr_polyline_fc(a, out_fc, oid_fld, shp_fld, SR):
    """Make polyline featureclass from a structured array."""
    f = obj_polyline(a, oid_fld, shp_fld, SR)
    return shapes_fc(f, out_fc)


def arr_polygon_fc(a, out_fc, oid_fld, shp_fld, SR):
    """Make polyline featureclass from a structured array."""
    f = obj_polygon(a, oid_fld, shp_fld, SR)
    return shapes_fc(f, out_fc)


def shapes_fc(shps, out_fc):
    """Create a featureclass/shapefile from poly* shapes.

    Requires:
    ---------

    - shps : geometry
        Point geometry objects needed to create the featureclass
    - out_fc : string
        Full path and name to the output container (gdb or folder)
    """
    msg0 = "\nCreated..\n{}".format(out_fc)
    msg1 = "\nCan't overwrite the {}... rename it".format(out_fc)
    arcpy.overwriteOutput = True
    try:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(shps, out_fc)
        tweet(msg0)
    except ValueError:
        tweet(msg1)


# ---- ******* main conversion section ***** ----
# ----         ----------------------        ----

def _id_geom_array(in_fc):
    """The main code segment which gets the id and shape information and
    explodes the geometry to individual points
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    a_flds = [oid_fld, shp_fld]
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, field_names=a_flds,
                                          explode_to_points=True,
                                          spatial_reference=SR)
    dt = [('Idx', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    a.dtype = dt
    return a


def polylines_arr(in_fc, as_struct=True):
    """Create an array from polylines.
    """
    return _id_geom_array(in_fc)


def polygons_arr(in_fc, as_struct=True, shp_fld=None, SR=None):
    """Create an array from polygons.
    """
    return _id_geom_array(in_fc)


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
    fld_names = [f.name for f in flds_oth]
    oid_geom = [oid_fld, 'SHAPE@X', 'SHAPE@Y']
    nulls = {'Double':np.nan,
             'Single':np.nan,
             'Short':np.iinfo(np.int16).min,
             'SmallInteger':np.iinfo(np.int16).min,
             'Long':np.iinfo(np.int32).min,
             'Float':np.nan,
             'Integer':np.iinfo(np.int32).min,
             'String':str(None),
             'Text':str(None)}
    fld_dict = {i.name: i.type for i in flds_oth}
    null_dict = {f:nulls[fld_dict[f]] for f in fld_names}
    if flds == "":                        # return just OID and Shape values
        out_flds = oid_geom               # FID and Shape X, Y
    elif flds == "*":                     # all fields
        out_flds = oid_geom + fld_names
    else:
        out_flds = [oid_fld, 'SHAPE@X', 'SHAPE@Y']
        for f in flds_oth:
            if f.name in flds:
                out_flds.append(f.name)
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
    return a


def change_fld(flds):
    """Convert the field types to array friendly ones
    """
    info = [(fld.type, fld.name, fld.length) for fld in flds]
    dt = []
    for i in info:
        if i[0] in ('OID', 'Integer', 'Long', 'Short', 'SmallInteger'):
            dt.append((i[1], '<i4'))
        elif i[0] in ('Double', 'Single', 'Float'):
            dt.append((i[1], '<f8'))
        else:
            dt.append((i[1], "{}{}".format('U', i[2])))
    return dt


def tbl_arr(in_fc):
    """ Convert a table or featureclass table (in_fc)to a numpy array
    including the oid field but excluding the geometry field.
    """
    desc = arcpy.da.Describe(in_fc)  # use the new da.Describe method
    dump = ['shapeFieldName', 'areaFieldName', 'lengthFieldName']
    f_geo = [desc[i] for i in dump]
    fields = [f for f in arcpy.ListFields(in_fc) if f.name not in f_geo]
    f_names = [f.name for f in fields]
    dt = change_fld(fields)
    vals = []
    with arcpy.da.SearchCursor(in_fc, field_names=f_names) as rows:
        for row in rows:
            vals.append(row)
        del row
        del rows
    dt = [(i.replace('OBJECTID', 'Idx'), j) for i, j in dt]
    a = np.asarray(vals, dtype=dt)
    return a  # vals, az


# ---- functions to convert between array and featureclass ----
#
def to_fc(out_fc, a, b=None, dim=2, flds=['Id', 'X', 'Y'], SR_code=None):
    """Reconstruct a featureclass from a deconstructed pair of arrays.

    This function reverses the functionality of to_array which splits a
    featureclass into an array of geometry and one of attributes.

    One can perform operations on one or the other or both, then reassemble
    into a new file.

    Requires:
    --------
    `out_fc` : filename
        full path and name for the output featureclass
    `a` : geometry array
        the array of geometry objects
    `b` : attribute array
        the array of attributes for 'a'
    `dim` : geometry type
        - 0 (point)
        - 1 (polyline) or
        - 2 (polygon)
    `fld` : id and shape field(s)
        normally ['Id, 'X', 'Y'] if deconstructed using 'to_array'
    `SR_code` :
        The spatial reference code of the output geometry

        - 4326 for GCS WGS84
        - 2951 for MTM 9

    References:
    ----------

        Spatial reference http://spatialreference.org/
    """
    args = [to_fc.__module__, dedent(to_fc.__doc__)]
    msg = "\n...to_fc ... in {} failed\n{}".format(*args)
    try:
        SR = arcpy.SpatialReference(SR_code)
    except ValueError:
        tweet("Spatial reference is in error")
        tweet(msg)
        return None
    if (dim not in (0, 1, 2)) or (len(flds) not in (2, 3)):
        tweet(msg)
        return None
    if len(flds) == 2:
        oid_fld, shp_fld = flds
    else:
        oid_fld, shp_fld = flds[0], flds[1:3]
    if dim == 0:
        arr_pnts(a, out_fc, shp_fld, SR)
#    geom = _split_array(a, fld=oid_fld)
    geom = np.split(a, np.where(np.diff(a[oid_fld]))[0] + 1)
    prts = [i[['Xs', 'Ys']].tolist() for i in geom]
    if dim == 1:
        arr_polyline_fc(geom, out_fc, oid_fld, shp_fld, SR)
    else:
        # arr_polygon_fc(geom, out_fc, oid_fld, shp_fld, SR)
        # pts = _arr_common(a, oid_fld, shp_fld)
        f = []
        for pt in prts:  # g
            f.append(arcpy.Polygon(
                             arcpy.Array([arcpy.Point(*p) for p in pt]), SR))
#    _shapes_fc(f, out_fc)
#    arcpy.da.ExtendTable(out_fc, table_match_field=oid_fld,
#                         in_array=b, array_match_field=oid_fld)
    return f


# ----------------------------------------------------------------------
# __main__ .... code section

if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    from _common import fc_info, tweet
    # from arcpytools import array_fc, array_struct, tweet
#    print("Script... {}".format(script))
#    _demo()
#    in_fc0 = r"C:\Git_Dan\a_Data\testdata.gdb\Points_10"
#    in_fc1 = r"C:\Git_Dan\a_Data\testdata.gdb\Polyline_connected"
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5km"   # full 25 polygons
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\polygon"
#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_0_big_3"
#    in_fc = r"C:\Data\Canada\CAN_adm0.gdb\CAN_0_sp"
#    a0, _ = two_arrays(in_fc, both=False, split=False)

#    fc_array(in_fc1, flds="", allpnts=True)
#    out_fc = r"C:\Git_Dan\a_Data\testdata.gdb\test_1"
#    out_tbl = r"C:\Git_Dan\a_Data\testdata.gdb\join_tbl"
