# -*- coding: UTF-8 -*-
"""
angles
======

Script:   angles.py

Author:   Dan.Patterson@carleton.ca

Modified: 2018-03-31

Purpose:  tools for working with numpy arrays

Useage:

References:
-----------
[1] https://en.wikipedia.org/wiki/Regular_polygon

Notes:
------
sum of interior angles   (n-2) * 180, where n is the number of sides

n = 3  180 triangle

n = 4  360 rectangle

n = 5  540 pentagram

n = 6  720 hexagram``
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
import arcpy


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- arcpytools functions --------------------------------------------------
# ---- You can remove these and import `arcpytools` ----
# ------------------------------------------------------
def _describe(in_fc):
    """Simply return the arcpy.da.Describe object

    *desc.keys()* : an abbreviated list...
    ::
        'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
        ... 'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
        'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
        'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
        'spatialReference',  ...
    """
    return arcpy.da.Describe(in_fc)


def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including...

    Parameters
    ----------
    - ``shp_fld  :``
        field name which contains the geometry object
    - ``oid_fld  :``
        the object index/id field name
    - ``SR       :``
        spatial reference object (use SR.name to get the name)
    - ``shp_type :``
        shape type (Point, Polyline, Polygon, Multipoint, Multipatch)
    - ``others   :``
        areaFieldName, baseName, catalogPath, featureType, fields,
        hasOID, hasM, hasZ, path
    - ``all_flds :``
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


def tweet(msg):
    """Produce a message for both arcpy and python::

    `msg` - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


# ---- end of `arcpytools` functions -----------------------------------------
#

def _geo_array(polys):
    """Convert polygon objects to arrays
    """
    arrays = [np.asarray(pt.__geo_interface__['coordinates']).squeeze()
              for pt in polys]  # for p in pt]
    return arrays


def _get_shapes(in_fc):
    """Get shapes from a featureclass, in_fc, using SHAPE@ returning
    :  [<Polygon object at....>, ... (<Polygon object at....>]
    """
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@') as cursor:
        a = [row[0] for row in cursor]
    return a


def arcpnts_poly(in_, out_type='Polygon', SR=None):
    """Convert arcpy Point lists to poly* features
    : out_type - either 'Polygon' or 'Polyline'
    :
    """
    s = []
    for i in in_:
        arr = arcpy.Array(i)
        if out_type == 'Polyline':
            g = arcpy.Polyline(arr, SR)
        elif out_type == 'Polygon':
            g = arcpy.Polygon(arr, SR)
        elif out_type == 'Points':
            g = arcpy.arcpy.Multipoint(arr[0], SR)
        s.append(g)
    return s


def angle_seq(a):
    """Sequential two point angle along a poly* feature
    :
    : angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x);
    : Accepted answer from the poly_angles link
    """
    frum = a[:-1]
    too = a[1:]
    diff = too - frum
    ang_ab = np.arctan2(diff[:, 1], diff[:, 0])
    return np.degrees(ang_ab)


def angles_poly(a, inside=True, in_deg=True):
    """Sequential angles from a poly* shape
    : a - an array of points, derived from a polygon/polyline geometry
    : inside - determine inside angles, outside if False
    : in_deg - convert to degrees from radians
    :
    :Notes:
    :-----
    : 2 points - subtract 2nd and 1st points, effectively making the
    :  calculation relative to the origin and x axis, aka... slope
    : n points - sequential angle between 3 points
    : - Check whether 1st and last points are duplicates.
    :   'True' for polygons and closed loop polylines, it is checked using
    :   np.allclose(a[0], a[-1])  # check first and last point
    : - a rolling tuple is constructed to produce the point triplets
    :   r = (-1,) + tuple(range(len(a))) + (0,)
    :   for np.arctan2(np.linalg.norm(np.cross(ba, bc)), np.dot(ba, bc))
    :
    :Reference:
    :---------
    : https://stackoverflow.com/questions/21483999/
    :         using-atan2-to-find-angle-between-two-vectors
    :  *** keep to convert object to array
    : a - a shape from the shape field
    : a = p1.getPart()
    : b =np.asarray([(i.X, i.Y) if i is not None else ()
    :                for j in a for i in j])
    """
    # a = a.getPart()
    # a = np.asarray([[i.X, i.Y] for j in a for i in j])
    if len(a) < 2:
        return None
    elif len(a) == 2:  # **** check
        ba = a[1] - a[0]
        return np.arctan2(*ba[::-1])
    else:
        angles = []
        dx, dy = a[0] - a[-1]
        if np.allclose(dx, dy):  # closed loop
            a = a[:-1]
            r = (-1,) + tuple(range(len(a))) + (0,)
        else:
            r = tuple(range(len(a)))
        for i in range(len(r)-2):
            p0, p1, p2 = a[r[i]], a[r[i+1]], a[r[i+2]]
            ba = p1 - p0
            bc = p1 - p2
            cr = np.cross(ba, bc)
            dt = np.dot(ba, bc)
            ang = np.arctan2(np.linalg.norm(cr), dt)
            angles.append(ang)
    if in_deg:
        angles = np.degrees(angles)
    return angles


def call_angles(a):
    """Call angles for each shape
    """
    out = []
    for i in a:
        out.append(angles_poly(i, inside=True, in_deg=True))
    return out


def prn_report(arrs, out):
    """Print a report summarizing the output
    """
    hdr = """
    :----------------------------------------------------------------------
    :Angle report....
    """
    frmt = """
    ({}) number of angles... ({})
    :array points...
    {}\n
    :interior angles {}
    :sum interior... {}
    """
    print(dedent("\n{}").format(hdr))
    cnt = 0
    for i in arrs:
        args = [cnt, len(i), i, out[cnt], sum(out[cnt])]
        prn = [str(i) for i in args]
        print(dedent(frmt).format(*prn))
        cnt += 1


# ------------------------------------------------------------------------
# (1) ---- Checks to see if running in test mode or from a tool ----------
def _demo():
    """run when script is in demo mode"""
    pth = script.replace('/angles.py', '')
    in_fc = 'C:/Git_Dan/a_Data/arcpytools_demo.gdb/polylines'
#    in_fc = 'C:/Git_Dan/a_Data/arcpytools_demo.gdb/three_shapes'
#    in_fc = pth + '/geom_data.gdb/three_shapes'
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5"   # full 25 polygons
#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\xy1000_tree"
    out_fc = pth + '/geom_data.gdb/x'
    out_type = 'Polygon'
    testing = True
    return in_fc, out_fc, out_type, testing


def _tool():
    """run when script is from a tool"""
    in_fc = sys.argv[1]  #
    out_fc = sys.argv[2]  #
    out_type = sys.argv[3]  # Polygon, Polyline are options
    testing = False
    return in_fc, out_fc, out_type, testing

# ---- main block ------------------------------------------------------------
#
# (1) check to see if in demo or tool mode
# (2) obtain fc information
# (3) split the fc into two arrays, one geometry, the 2nd attributes
# (4) obtain the shapes and densify
# (5) optionally produce the output fc


if len(sys.argv) == 1:
    in_fc, out_fc, out_type, testing = _demo()
else:
    in_fc, out_fc, out_type, testing = _tool()

shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)


# ---- produce output --------------------------------------------------------

polys = _get_shapes(in_fc)
arrs = _geo_array(polys)
out = call_angles(arrs)
# out = angles_poly(arrs, inside=True, in_deg=True)  # use for xy1000_tree only
# a0 = np.array([[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]])
# a1 = np.array([[20., 20.], [20., 30.], [30., 30.], [30., 20.], [20., 20.]])
# p0, p1, p2 = polys
# b = _get_attributes(in_fc)

# out_shps = arcpnts_poly(out, out_type=out_type, SR=SR)
# if not testing:
#     if arcpy.Exists(out_fc):
#         arcpy.Delete_management(out_fc)
#     arcpy.CopyFeatures_management(out_shps, out_fc)


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
