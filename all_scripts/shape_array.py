# -*- coding: UTF-8 -*-
"""
:Script:   shape_array.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-08-16
:Purpose:  Tools for working with arcpy geometry objects and conversion to
:          numpy arrays.
:Useage:
:
:Notes:
:-----
:(1) JSON --- example dictionary... a_polygon.JSON returns a string
:  json.loads(a_polygon.JSON)  # 3 polygons
:  {'rings': [[[300010, 5000010],... [300010, 5000010]],
:             [[300010, 5000010],... [300010, 5000010]],
:             [[300005, 5000008],... [300005, 5000008]]],
:   'spatialReference': {'latestWkid': 2951, 'wkid': 2146}}
:
:(2) __geo_interface__ --- example return for a 2-part polygon
:  a_polygon.__geo_interface__
:  {'coordinates': [[[(300010.0, 5000010.0),... (300010.0, 5000010.0)]],
:                   [[(300010.0, 5000010.0),... (300010.0, 5000010.0)],
:                    [(300005.0, 5000008.0),... (300005.0, 5000008.0)]]],
:   'type': 'MultiPolygon'}
:
:(3) JSON and WKT return strings, that is why you need json.loads or
:  __geo_interface__
:
:  JSON
:  a_polygon.JSON
:  '{'rings' ...snip... 'wkid': 2146}}'  # note the ' ' enclosure
:  WKT --- WKT returns a string like JSON
:  a_polygon.WKT
:  'MULTIPOLYGON(((300010 5000010,... 300010 5000010)),
:                 ((300010 5000010,... 300010 5000010),
:                  (300005 5000008,... 300005 5000008)))'
:
:(4) cursors .....
:  from arcgisscripting import da
:
:  dir(da)  # the main data access link with underscore functions present
:
:  ['Describe', 'Domain', 'Editor', 'ExtendTable', 'FeatureClassToNumPyArray',
:  'InsertCursor', 'ListDomains', 'ListFieldConflictFilters', 'ListReplicas',
:  'ListSubtypes', 'ListVersions', 'NumPyArrayToFeatureClass',
:  'NumPyArrayToTable', 'Replica', 'SearchCursor', 'TableToNumPyArray',
:  'UpdateCursor', 'Version', 'Walk', '__doc__', '__loader__', '__name__',
:  '__package__', '__spec__', '_internal_eq', '_internal_sd', '_internal_vb']
:
:  dir(da.SearchCursor)
:
:  ['__class__', '__delattr__', '__dir__', '__doc__', '__enter__', '__eq__',
:  '__esri_toolinfo__', '__exit__', '__format__', '__ge__', '__getattribute__',
:  '__getitem__', '__gt__', '__hash__', '__init__', '__iter__', '__le__',
:  '__lt__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__',
:  '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
:  '_as_narray', '_dtype', 'fields', 'next', 'reset']


:(5) arcpy.da.SearchCursor
:  cur = arcpy.da.SearchCursor(in_table, field_names, {where_clause},
:                             {spatial_reference}, {explode_to_points},
:                             {sql_clause})
:  field_names
:    - flds = [i.name for i in arcpy.ListFields(in_fc)]  # fields, sort after
:    - flds = "*"  # all fields in order
:    - flds = ['OBJECTID', 'Shape',...]  # specify the fields you want
:  where_clause
:    - specify a where clause
:  spatial_reference
:    - SR.name    'NAD_1983_CSRS_MTM_9'
:    - SR.PCSName 'NAD_1983_CSRS_MTM_9'
:    - SR.PCSCode 2951
:  explode_to_points
:    - True or False
:  sql_clause
:    - specify one or (None, None)
:
:  For example....
:    args = [in_fc, ['OBJECTID', 'Shape'], None, None,  True, (None, None)]
:    cur = arcpy.da.SearchCursor(*args)
:    a = cur._as_narray()
:
:Timing tests:
:------------
:  a.shape => (1814562, 2) 1,814,462 points, from 3 multipart arrays
:
:  a = _cursor_array(in_fc, full=True)  Time: 4.68e+01s for 3 objects
:  a = _cursor_array(in_fc, full=False) Time: 2.29e+00s for 1,814,562 objects
:
:  a = _cursor_shp(in_fc, full=True)    Time: 1.26e+00s for 3 objects
:  a = _cursor_shp(in_fc, full=False)   Time: 8.22e-01s for 3 objects
:
:  b = _geo_array(a)
:
: Timing function for... _geo_array
:   Time: 4.50e+01s for 3 objects
:
:Notes:
:Polygons
: information - __doc__, __module__, __type_string__
: conversion - JSON, WKB, WKT, __geo_interface__, _fromGeoJson
: properties - 'area', 'boundary', 'centroid', 'convexHull', 'equals',
:    'firstPoint, 'extent', 'isMultipart', 'hullRectangle', 'labelPoint',
:    'lastPoint', 'length', 'length3D', 'partCount', 'pointCount',
:    'spatialReference', 'trueCentroid', 'type'
:
: methods - angleAndDistanceTo, buffer, clip, contains, crosses, cut, densify,
:    difference, disjoint, distanceTo, generalize, getArea, getGeohash,
:    getLength, getPart, intersect, measureOnLine, overlaps,
:    pointFromAngleAndDistance, positionAlongLine, projectAs,
:    queryPointAndDistance, segmentAlongLine, snapToLine, symmetricDifference,
:    touches, union, within
:
:
:References:
:----------
: arcpy.da.SearchCursor
:   http://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import os
from textwrap import dedent, indent
import numpy as np
import arcpy

from arraytools import fc_info, time_deco
import arraytools as art

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=80, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__imports__ = ['time_deco',
               'fc_info']


# ----------------------------------------------------------------------
# ---- extras ----


def flatten(a_list, flat_list=None):
    """Change the isinstance as appropriate
    :  Flatten an object using recursion
    :  see: itertools.chain() for an alternate method of flattening.
    """
    if flat_list is None:
        flat_list = []
    for item in a_list:
        if isinstance(item, (list, tuple, np.ndarray, np.void)):
            flatten(item, flat_list)
        else:
            flat_list.append(item)
    return flat_list


def flatten_shape(shp, completely=False):
    """Flatten a shape using itertools.
    : shp - shape, polygon, polyline, point
    : completely - True, returns points for all objects
    :            - False, returns Array for polygon or polyline objects
    : Notes:
    :------
    : __iter__ - Polygon, Polyline, Array all have this property... Points
    :            do not.
    """
    import itertools
    if completely:
        vals = [i for i in itertools.chain(shp)]
    else:
        vals = [i for i in itertools.chain.from_iterable(shp)]
    return vals


def unpack(iterable, param='__iter__'):
    """Unpack an iterable based on the param(eter) condition using recursion.
    :Notes:
    : - Use 'flatten' for recarrays or structured arrays'
    : ---- see main docs for more information and options ----
    : To produce uniform array from this, use the following after this is done.
    :   out = np.array(xy).reshape(len(xy)//2, 2)
    : isinstance(x, (list, tuple, np.ndarray, np.void)) like in flatten above
    """
    xy = []
    for x in iterable:
        if hasattr(x, '__iter__'):
            xy.extend(unpack(x))
        else:
            xy.append(x)
    return xy
# ----------------------------------------------------------------------
# list files in folder
# import os
# d = path  # the file/folder path
# [os.path.join(d,o) for o in os.listdir(d)
#                    if os.path.isdir(os.path.join(d,o))]


def get_dir(path):
    """Get the directory list from a path, excluding geodatabase folders
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    p = os.path.normpath(path)
    full = [os.path.join(p, v) for v in os.listdir(p)]
    dirlist = [val for val in full if os.path.isdir(val)]
    dirlist.sort()
    return dirlist


def print_folders(path, first=True, prefix=""):
    """ Print recursive listing of contents of path """
    if first:  # Detect outermost call, print a heading
        print("Folder listing for....\n.... {}".format(path))
        prefix = "|  "
        first = False
    dirlist = get_dir(path)
    cp = os.path.commonprefix(dirlist)
#    print("common prefix {}".format(cp))
    for d in dirlist:
        fullname = os.path.join(path, d)   # Turn name into full pathname
        if os.path.isdir(fullname):        # If a directory, recurse.
            n = fullname.replace(cp, '.'*len(cp) + '\\')
            # print(prefix + "- " + fullname)
            print(prefix + "- " + n)  # fullname) # os.path.relpath(fullname))
            p = "   "
            print_folders(fullname, first=False, prefix=p)


# ---- Cursor functions -----------------------------------------------------
#
@time_deco
def _to_ndarray(in_fc, flds=None, SR=None, to_pnts=True):
    """Convert searchcursor shapes an ndarray quickly.
    :
    :Requires:
    : in_fc - input featureclass
    : SR - spatial reference, or WKID
    :Notes:
    :-----
    :  field_names
    :    ['OBJECTID', 'Shape'], ['OID@', 'Shape'], ['OID@', 'SHAPE@WKT']
    :    ['OID@', 'SHAPE@JSON'], ['OID@', 'SHAPE@X', 'SHAPE@Y']
    :  cur = arcpy.da.SearchCursor(in_fc, field_names, .....)
    :      =
    :  cur = arcpy.da.SearchCursor(in_fc, ['OBJECTID', 'Shape'], None, None,
    :                              True, (None, None))
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    cur = arcpy.da.SearchCursor(in_fc, flds, None, '2951', True, (None, None))
    flds = cur.fields
    dt = cur._dtype
    a = cur._as_narray()
    return a, flds, dt


def _arr_json(file_out, arr=None):
    """send an array out to json format
    :use json_arr to read
    :  no error checking
    """
    import json
    import codecs
    json.dump(arr.tolist(), codecs.open(file_out, 'w', encoding='utf-8'),
              sort_keys=True, indent=4)
    # ----


@time_deco
def get_geom(in_fc):
    """just get the geometry object"""
    coords = [np.asarray(row[0].__geo_interface__['coordinates'])
              for row in arcpy.da.SearchCursor(in_fc, ['SHAPE@'])]  # shape@
    # coords = [i.__geo_interface__['coordinates'] for i in geoms]
    return coords  # , g2


@time_deco
def _get_shapes(in_fc):
    """Get the shapes from a featureclass.
    :
    :Requires:
    :--------
    : in_fc - the featureclass, SHAPE@ is used to pull the full geometry object
    :
    :Returns:
    :-------
    :  A list of polygon objects in the form
    :  [<Polygon object at....>, ... (<Polygon object at....>]
    """
    a = []
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@') as cursor:  # "*"
        for row in cursor:
            a += row
    return a, cursor


def _props(a_shape, prn=True):
    """Get some basic shape geometry properties
    """
    coords = a_shape.__geo_interface__['coordinates']
    sr = a_shape.spatialReference
    props = ['type', 'isMultipart', 'partCount', 'pointCount', 'area',
             'length', 'length3D', 'centroid', 'trueCentroid', 'firstPoint',
             'lastPoint', 'labelPoint']
    props2 = [['Name', sr.name], ['Factory code', sr.factoryCode]]
    t = "\n".join(["{!s:<12}: {}".format(i, a_shape.__getattribute__(i))
                   for i in props])
    t = t + "\n" + "\n".join(["{!s:<12}: {}".format(*i) for i in props2])
    tc = '{!r:}'.format(np.array(coords))
    tt = t + "\nCoordinates\n" + indent(tc, '....')
    if prn:
        print(tt)
    else:
        return tt


@time_deco
def _geo_array(polys):
    """Convert the Polygon class, json to an array
    :
    """
    arrays = [np.array(pt.__geo_interface__['coordinates'])
              for pt in polys]  # for p in pt]
    return arrays


@time_deco
def _cursor_shp(in_fc, full=True):
    """Extract the point geometry from a featureclass
    :
    : in_fc - the featureclass
    : full - True: 'SHAPE@', False: ['SHAPE@X', 'SHAPE@Y' ]
    """
    shp = [['SHAPE@X', 'SHAPE@Y'], 'SHAPE@'][full]
    if full:
        a = [row[0].__geo_interface__['coordinates']
             for row in arcpy.da.SearchCursor(in_fc, shp)]
    else:
        a = [row for row in arcpy.da.SearchCursor(in_fc, shp,
                                                  explode_to_points=True)]
    return a


#@time_deco
def _cursor_array(in_fc, full=True):
    """Return the the points for a geometry object using a searchcursor.
    :
    : in_fc - the featureclass
    : full - True: 'SHAPE@', False: ['SHAPE@X', 'SHAPE@Y' ]
    """
    shp = [['SHAPE@X', 'SHAPE@Y'], 'SHAPE@'][full]
    if full:
        a = [np.asarray(row[0].__geo_interface__['coordinates'])
             for row in arcpy.da.SearchCursor(in_fc, shp)]
    else:
        a = [row for row in arcpy.da.SearchCursor(in_fc, shp,
                                                  explode_to_points=True)]
    a = np.array(a)
    return a


def _cross_3pnts(a):
    """Requires 3 points on a plane:
    """
    a = np.asarray(a)
    p0, p1, p2 = a
    u, v = a[1:] - a[0]  # p1 - p0, p2 - p0
    #u = unit_vector(u)
    #v = unit_vector(v)
    eq = np.cross(u, v)  # Cross product times one of the points
    d = sum(eq * p0)
    if d > 0.0:
        eq /= d
        d /= d
    else:
        d = 0.0
    return eq, d


def _demo():
    """
    : -
    """
    pass
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\polygon"
    # in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_0_big_3"
    # in_fc = r"C:\Data\Canada\CAN_adm0.gdb\CAN_0_sp"

    #a0 = [[0., 0., 0.,], [4., 0., 3.], [4., 3., 3.]]
    #a0 = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [-1.0, 2.0, 1.0]])
#    path = r'C:\Git_Dan'
#    pobj = [p for p in pathlib.Path(path).iterdir() if p.is_dir()]
#    pstr = [p._str for p in pathlib.Path(path).iterdir() if p.is_dir()]
    '''
    import pathlib
    p0 = pathlib.Path(r'C:\Temp\a\aa\a0\a00')
    p0 = 'C:/Git_Dan/arcpytools'
    p0._parts ... for a list or ...
    p0.parts  ... for tuple
    ... ('C:\\', 'Temp', 'a', 'aa', 'a0', 'a00')
    is_file, is_dir
    Out[33]: ['C:\\', 'Git_Dan', 'arcpytools']
    p0.parent

    p0.root    # '\\'
    p0.drive   # 'C:'
    p0.anchor  # 'C:\\'
    p0.stem    # 'a00'
    p0.parent  # WindowsPath('C:/Temp/a/aa/a0')

    os.walk(top[, topdown=True
    '''
