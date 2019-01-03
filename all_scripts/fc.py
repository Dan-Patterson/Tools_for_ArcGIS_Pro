# -*- coding: UTF-8 -*-
"""
fc.py  featureclass.py
======================

Script   :   fc.py  (featureclass.py)

Author   :   Dan_Patterson@carleton.ca

Modified : 2018-09-02

Purpose  :
    Tools for working with featureclass arcpy geometry objects and conversion
    to numpy arrays.

Notes:
------
- do not rely on the OBJECTID field for anything

  http://support.esri.com/en/technical-article/000010834

See ... /arcpytools/Notes_etc/fc_py_output.txt

for sample output for the functions below

-  _describe,  : arcpy describe object
-  _get_shapes,  : actual arc* geometry
-  _ndarray,  : a structured array
-  _props  : detailed properties for an arcpy shape object
-  _two_arrays,  : a geometry array and attribute array
-  _xy,  : x,y coordinates only
-  _xyID,  : x,y and ID
-  _xy_idx,  : x,y and an index array
-  change_fld,  : provide array happy field types
-  fc_info,  : shape and oid fields, SR and geometry type
-  fld_info,  : field type, namd and length

General
-------

cur._as_narray() and cur._dtype are methods of da.cursors

field_names
  ['OBJECTID', 'Shape'], ['OID@', 'Shape'], ['OID@', 'SHAPE@WKT']
  ['OID@', 'SHAPE@JSON'], ['OID@', 'SHAPE@X', 'SHAPE@Y']

>>> cur = arcpy.da.SearchCursor(in_fc, field_names, .....)
>>> cur = arcpy.da.SearchCursor(in_fc, ['OBJECTID', 'Shape'], None, None,
                                True, (None, None))

Polygons
--------

Arcpy polygon objects

- information :
    __doc__, __module__, __type_string__

- conversion :
    JSON, WKB, WKT, __geo_interface__, _fromGeoJson

- properties :
    'area', 'boundary', 'centroid', 'convexHull', 'equals', 'firstPoint,
    'extent', 'isMultipart', 'hullRectangle', 'labelPoint', 'lastPoint',
    'length', 'length3D', 'partCount', 'pointCount', 'spatialReference',
    'trueCentroid', 'type'

- methods :
    angleAndDistanceTo, buffer, clip, contains, crosses, cut, densify,
    difference, disjoint, distanceTo, generalize, getArea, getGeohash,
    getLength, getPart, intersect, measureOnLine, overlaps,
    pointFromAngleAndDistance, positionAlongLine, projectAs,
    queryPointAndDistance, segmentAlongLine, snapToLine, symmetricDifference,
    touches, union, within


**1. cursors**

from arcgisscripting import da

dir(da)  : the main data access link with underscore functions present

   ['Describe', 'Domain', 'Editor', 'ExtendTable', 'FeatureClassToNumPyArray',
   'InsertCursor', 'ListDomains', 'ListFieldConflictFilters', 'ListReplicas',
   'ListSubtypes', 'ListVersions', 'NumPyArrayToFeatureClass',
   'NumPyArrayToTable', 'Replica', 'SearchCursor', 'TableToNumPyArray',
   'UpdateCursor', 'Version', 'Walk', '__doc__', '__loader__', '__name__',
   '__package__', '__spec__', '_internal_eq', '_internal_sd', '_internal_vb']

dir(da.SearchCursor)

   ['__class__', '__delattr__', '__dir__', '__doc__', '__enter__', '__eq__',
   '__esri_toolinfo__', '__exit__', '__format__', '__ge__', '__getattribute__',
   '__getitem__', '__gt__', '__hash__', '__init__', '__iter__', '__le__',
   '__lt__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__',
   '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
   '_as_narray', '_dtype', 'fields', 'next', 'reset']


**2. arcpy.da.SearchCursor**

>>> cur = arcpy.da.SearchCursor(in_table, field_names, {where_clause},
                                {spatial_reference}, {explode_to_points},
                                {sql_clause})
   - field_names
     - flds = [i.name for i in arcpy.ListFields(in_fc)]  : fields, sort after
     - flds = '*'  : all fields in order
     - flds = ['OBJECTID', 'Shape',...]  : specify the fields you want
   - where_clause
     - specify a where clause
   - spatial_reference
     - SR.name    'NAD_1983_CSRS_MTM_9'
     - SR.PCSName 'NAD_1983_CSRS_MTM_9'
     - SR.PCSCode 2951
   - explode_to_points
     - True or False
   - sql_clause
     - specify one or (None, None)

For example....

>>> args = [in_fc, ['OBJECTID', 'Shape'], None, None,  True, (None, None)]
>>> cur = arcpy.da.SearchCursor(*args)
>>>  a = cur._as_narray()

**3. JSON** --- example dictionary... a_polygon.JSON returns a string
::
   json.loads(a_polygon.JSON)  : 3 polygons
   {'rings':  [[[300010, 5000010],... [300010, 5000010]],
              [[300010, 5000010],... [300010, 5000010]],
              [[300005, 5000008],... [300005, 5000008]]],
    'spatialReference': {'latestWkid': 2951, 'wkid': 2146}}

**4. __geo_interface__**--- example return for a 2-part polygon
::
   a_polygon.__geo_interface__
   {'coordinates': [[[(300010.0, 5000010.0),... (300010.0, 5000010.0)]],
                    [[(300010.0, 5000010.0),... (300010.0, 5000010.0)],
                     [(300005.0, 5000008.0),... (300005.0, 5000008.0)]]],
    'type': 'MultiPolygon'}


**5. JSON and WKT** return strings, that is why you need json.loads or
::
   __geo_interface__
   a_polygon.JSON
   '{'rings' ...snip... 'wkid': 2146}}'  : note the ' ' enclosure
   WKT --- WKT returns a string like JSON
   a_polygon.WKT
   'MULTIPOLYGON(((300010 5000010,... 300010 5000010)),
                  ((300010 5000010,... 300010 5000010),
                   (300005 5000008,... 300005 5000008)))'

Other examples:
--------------

The following returned objects for each approach are:
    p0 - list
    p1 - ndarray
    p2a, p2b = tuple of ndarrays
    p3 - ndarray
    p4 - ndarray

>>> p0 = [i.__geo_interface__['coordinates'] for i in in_polys]
>>> p1 = _xyID(in_fc)
>>> p2a, p2b = _xy_idx(in_fc)
>>> p3 = _xy(in_fc)
>>> p4 = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                           ["OID@", "Shape@X", "Shape@Y"],
                                           explode_to_points=True,
                                           spatial_reference=SR)

geometry is a square one of 5 shapes

>>> p0[2]  # __geo_interface__
[[[(307500.0, 5029300.0),
   (308786.7818999998, 5029300.0),
   (308792.1923000002, 5028500.0),
   (307500.0, 5028500.0),
   (307500.0, 5029300.0)]]]

>>> p1[p1['IDs'] == 3]
array([(3, 307500.  , 5029300.), (3, 308786.78, 5029300.),
       (3, 308792.19, 5028500.), (3, 307500.  , 5028500.),
       (3, 307500.  , 5029300.)],
      dtype=[('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')])

>>> p2a, p2b = _xy_idx(in_fc)
>>> idx = p2b[:,0] == 3
>>> p2a[idx]
array([[ 307500.  , 5029300.  ],
       [ 308786.78, 5029300.  ],
       [ 308792.19, 5028500.  ],
       [ 307500.  , 5028500.  ],
       [ 307500.  , 5029300.  ]])

>>> p3  # just the coordinates regardless of the poly*

>>> p4  # arcpy.da.FeatureClassToNumPyArray(......)
>>> p4[p4['OID@'] == 3]
array([(3, 307500.   , 5029300.), (3, 308786.782, 5029300.),
       (3, 308792.192, 5028500.), (3, 307500.   , 5028500.),
       (3, 307500.   , 5029300.)],
      dtype=[('OID@', '<i4'), ('Shape@X', '<f8'), ('Shape@Y', '<f8')])

Timing
  p0  2.31 ms ± 27.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
  p1  5 ms ± 42.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
  p2  4.88 ms ± 143 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
  p3  4.93 ms ± 135 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
  p4  4.85 ms ± 110 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
References:
----------

arcpy.da.SearchCursor

http://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm
  ---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
from textwrap import indent
import numpy as np
import arcpy

import warnings
warnings.simplefilter('ignore', FutureWarning)

from arraytools.fc_tools._common import fc_info, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=3, suppress=True,
                    threshold=50, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['_cursor_array',
           '_geo_array',
           '_get_shapes',
           '_ndarray',
           '_two_arrays',
           '_xy', '_xyID', '_xy_idx',
           'orig_dest_pnts',
           'obj_array',
           'change_fld',
           '_props',
           'join_arr_fc'
           ]


# ----------------------------------------------------------------------------
# ---- Functions to get geometry and attributes ------------------------------
# ----------------------------------------------------------------------------
# (1) To ndarray/structured/recarry
#
def _cursor_array(in_fc, full=True):
    """Return the the points for a geometry object using a searchcursor.

    in_fc :
        the featureclass
    full :
        True: 'SHAPE@', False: ['SHAPE@X', 'SHAPE@Y' ]
    """
    shp = [['SHAPE@X', 'SHAPE@Y'], 'SHAPE@'][full]
    if full:
        a = [np.asarray(row[0].__geo_interface__['coordinates'])
             for row in arcpy.da.SearchCursor(in_fc, shp)]
    else:
        a = [row for row in arcpy.da.SearchCursor(in_fc, shp,
                                                  explode_to_points=True)]
    a = np.asarray(a).squeeze()
    return a


def _geo_array(polys):
    """Convert polygon objects to arrays
    """
    arrays = [np.asarray(pt.__geo_interface__['coordinates']).squeeze()
              for pt in polys]  # for p in pt]
    return arrays


def _get_shapes(in_fc):
    """Get shapes from a featureclass, in_fc, using SHAPE@ returning
       [<Polygon object at....>, ... (<Polygon object at....>]
    """
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@') as cursor:
        a = [row[0] for row in cursor]
    return a


def _ndarray(in_fc, to_pnts=True, flds=None, SR=None):
    """Convert featureclass geometry (in_fc) to a structured ndarray including
    options to select fields and specify a spatial reference.

    Requires
    --------
    in_fc : string
        input featureclass
    to_pnts : boolean
        True, convert the shape to points. False, centroid returned.
    flds : string or list of strings
      - '*' for all
      - others : 'OID@', 'Shape',  ['SHAPE@X', 'SHAPE@Y'], or specify
    Note:
    -----
    You cannot use the 'SHAPE@' field
    Example:
    --------
    a = _ndarray(in_fc, True, ['OID@',' SHAPE@X', 'SHAPE@Y', None]
    """
    if flds is None:
        bad = ['OID', 'Geometry', 'Shape_Length', 'Shape_Area']
        f0 = ["OID@", "SHAPE@X", "SHAPE@Y"]
        f1 = [i.name for i in arcpy.ListFields(in_fc)
              if i.type not in bad]
        flds = f0 + f1
    if SR is None:
        desc = arcpy.da.Describe(in_fc)
        SR = desc['spatialReference']
    args = [in_fc, flds, "", SR, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    del cur
    return a


def _two_arrays(in_fc, both=True, split=True):
    """Send to a numpy structured/array and split it into a geometry array
    and an attribute array.  They can be joined back later if needed.

    Note
    ----
        The geometry array is returned as an object array.  See the
        main documentation

    Requires:
    --------

    functions:
        _xyID
            function to get geometry array
        _ndarray
            function to get the x, y, id array and attribute array
        fc_info(in_fc)
            function needed to return fc properties
    parameters:
        both
            True, to return both arrays, False to return just geometry
        split
            True, split points by their geometry groups as an object array;
            False, a sequential array with shape = (N,)
    variables:

    >>> dt_a = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    >>> dt_b = [('IDs', '<i4'), ('Xc', '<f8'), ('Yc', '<f8')]
    >>> dt_b.extend(b.dtype.descr[2:])

        Extend the dtype using the attribute dtype minus geometry and id
    """
    a = _xyID(in_fc, to_pnts=True)
    shp_fld, oid_fld, SR, shp_type = fc_info(in_fc)
    dt_a = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    dt_b = [('IDs', '<i4'), ('Xc', '<f8'), ('Yc', '<f8')]
    a.dtype = dt_a
    b = None
    if split:
        ids = np.unique(a['IDs'])
        w = np.where(np.diff(a['IDs']))[0] + 1
        a = np.split(a, w)
        a = np.array([[ids[i], a[i][['Xs', 'Ys']]] for i in range(len(ids))])
    if both:
        b = _ndarray(in_fc, to_pnts=False, flds=None, SR=None)
        dt_b.extend(b.dtype.descr[2:])
        b.dtype = dt_b
    return a, b


def _xy(in_fc):
    """Convert featureclass geometry (in_fc) to a simple 2D point array.
    See _xyID if you need id values.
    """
    flds = ['SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, True, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    N = len(a)
    a = a.view(dtype='float64')
    a = a.reshape(N, 2)
    a = np.copy(a, order='C')
    del cur
    return a


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


def _xy_idx(in_fc):
    """Convert featureclass geometry (in_fc) to a simple 2D point array with
    float64 data type and a separate index array to preserve id values
    ***Best version comparied to _two_arrays
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, True, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    N = len(a)
    idx = np.zeros((N, 2), dtype='int')
    idx[:, 0] = a['OID@']
    id_n = np.cumsum(np.bincount(idx[:, 0]))
    diff = np.diff(id_n, n=1)
    s = [np.arange(i) for i in diff]
    idx[:, 1] = np.hstack(s)
    a = a[['SHAPE@X', 'SHAPE@Y']]
    a = a.view(dtype='float64')
    a = a.reshape(N, 2)
    del cur
    return a, idx


def orig_dest_pnts(fc, SR):
    """Convert sequential points to origin-destination pairs to enable
    construction of a line.

    Notes:
    -----
    a : array

    >>> arcpy.da.FeatureClassToNumPyArray(
            in_table=fc,
            field_names=["OID@","Shape@X", "Shape@Y"],
            where_clause=None
            spatial_reference="2951")
            explode_to_points=False, skip_nulls=False,
            null_value=None, sql_clause=(None, None)

    out : from_to_pnts(fc)

    >>> arcpy.da.ExtendTable(
         in_table=fc,
         table_match_field='OBJECTID',  # normally
         in_array = out,                # the array you created
         array_match_field='IDs',       # created by this script
         append_only=True)

    Sample:
        fc = 'C:/Git_Dan/a_Data/arcpytools_demo.gdb/polylines_pnts'

        SR = '2951'
    """
    a = arcpy.da.FeatureClassToNumPyArray(fc, ["OID@", "Shape@X", "Shape@Y"],
                                          spatial_reference=SR)
    in_names = list(a.dtype.names)
    kinds = ['<i4', '<f8', '<f8', '<f8', '<f8']
    out_names = ['IDs', 'X_0', 'Y_0', 'X_1', 'Y_1']
    dt = list(zip(out_names, kinds))
    out = np.zeros(a.shape[0], dtype=dt)
    arrs = [a[i] for i in in_names]
    X_t = np.roll(a[in_names[-2]], -1)
    Y_t = np.roll(a[in_names[-1]], -1)
    arrs.append(X_t)
    arrs.append(Y_t)
    for i in range(len(out_names)):
        out[out_names[i]] = arrs[i]
    return out


def obj_array(in_fc):
    """Convert an featureclass geometry to an object array.
    The array must have an ID field.  Remove any other fields except
    IDs, Xs and Ys or whatever is used by the featureclass.

    Requires
    --------
        _xyID and a variant of group_pnts
    """
    def _group_pnts_(a, key_fld='IDs', shp_flds=['Xs', 'Ys']):
        """see group_pnts in tool.py"""
        returned = np.unique(a[key_fld], True, True, True)
        uniq, idx, inv, cnt = returned
        from_to = list(zip(idx, np.cumsum(cnt)))
#        from_to = [[idx[i-1], idx[i]] for i in range(1, len(idx))]
        subs = [a[shp_flds][i:j] for i, j in from_to]
        groups = [sub.view(dtype='float').reshape(sub.shape[0], -1)
                  for sub in subs]
        return groups
    #
    a = _xyID(in_fc)
    a_s = _group_pnts_(a, key_fld='IDs', shp_flds=['Xs', 'Ys'])
    a_s = np.asarray(a_s)
    return a_s


def change_fld(flds):
    """Convert the field types to array friendly ones.
    """
    info = [(fld.type, fld.name, fld.length) for fld in flds]
    dt = []
    for i in info:
        if i[0] in ('OID', 'Integer', 'Long', 'Short'):
            dt.append((i[1], '<i4'))
        elif i[0] in ('Double', 'Single', 'Float'):
            dt.append((i[1], '<f8'))
        else:
            dt.append(i[1], "{}{}".format('U', i[2]))
    return dt


def _props(a_shape, prn=True):
    """Get some basic shape geometry properties.

    Note:
    ----
        `a_shape`, is a single shape.
        A searchcursor will return a list of geometries, so you should slice
        even if there is only one shape.
    """
    if not hasattr(a_shape, '__geo_interface__'):
        tweet("Requires a 'shape', your provided a {}".format(type(a_shape)))
        return None
    coords = a_shape.__geo_interface__['coordinates']
    SR = a_shape.spatialReference
    props = ['type', 'isMultipart', 'partCount', 'pointCount', 'area',
             'length', 'length3D', 'centroid', 'trueCentroid', 'firstPoint',
             'lastPoint', 'labelPoint']
    props2 = [['SR Name', SR.name], ['SR Factory code', SR.factoryCode]]
    t = "\n".join(["{!s:<12}: {}".format(i, a_shape.__getattribute__(i))
                   for i in props])
    t = t + "\n" + "\n".join(["{!s:<12}: {}".format(*i) for i in props2])
    tc = '{!r:}'.format(np.array(coords))
    tt = t + "\nCoordinates\n" + indent(tc, '....')
    if prn:
        print(tt)
    else:
        return tt


# (11)_join_array ... code section .....
def join_arr_fc(a, in_fc, out_fld='Result_', OID_fld='OID@'):
    """Join an array to a featureclass table using matching fields, usually
    an object id field.

    Requires:
    --------

    a :
        an array of numbers or text with ndim=1
    out_fld :
        field name for the results
    in_fc :
        input featureclass
    in_flds :
        list of fields containing the OID@ field as a minimum

    ExtendTable (in_table, table_match_field,
                 in_array, array_match_field, {append_only})

    """
    N = len(a)
    dt_a = [('IDs', '<i4'), (out_fld, a.dtype.str)]
    out = np.zeros((N,), dtype=dt_a)
    out['IDs'] = [row[0] for row in arcpy.da.SearchCursor(in_fc, OID_fld)]
    out[out_fld] = a
    arcpy.da.ExtendTable(in_fc, OID_fld, out, 'IDs', True)
    return out


# ----------------------------------------------------------------------
#
def _cross_3pnts(a):
    """Requires 3 points on a plane:
    """
    a = np.asarray(a)
    p0, p1, p2 = a
    u, v = a[1:] - a[0]  # p1 - p0, p2 - p0
    # u = unit_vector(u)
    # v = unit_vector(v)
    eq = np.cross(u, v)  # Cross product times one of the points
    d = sum(eq * p0)
    if d > 0.0:
        eq /= d
        d /= d
    else:
        d = 0.0
    return eq, d


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
      - print the script source name.
      - run the _demo
    """
    from _common import fc_info, tweet
#    print("Script... {}".format(script))
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\square2"
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5km"   # full 25 polygons
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Polygon"
#    in_fc = r'C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_geom_sp_LCC'
#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_0_big_3"
#    in_fc = r"C:\Data\Canada\CAN_adm0.gdb\CAN_0_sp"
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Polygon_pnts"
#    flds = ['OBJECTID', 'Text_fld']
#    oid, vals = flds[0], flds[1:]
#    arr = arcpy.da.TableToNumPyArray(in_fc, flds)
#
#    in_tbl = r"C:\Git_Dan\arraytools\Data\numpy_demos.gdb\sample_10k"
#    in_flds = ['OBJECTID', 'County', 'Town']
#    a = arcpy.da.TableToNumPyArray(in_tbl, in_flds)
#    c = concat_flds(a['County'], a['Town'], sep='...', name='Locale')
#    c_id = np.zeros((len(c), ), dtype=[('IDs', '<i8')])
#    c_id['IDs'] = np.arange(1, len(c) + 1)
