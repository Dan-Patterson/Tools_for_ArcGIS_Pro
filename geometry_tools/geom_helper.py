# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- array functions -------------------------------------------------------
#

__all__ = ['tweet', '_describe', 'fc_info', '_xyID', '_ndarray', '_two_arrays']

def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


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
    :
    : shp_fld  - field name which contains the geometry object
    : oid_fld  - the object index/id field name
    : SR       - spatial reference object (use SR.name to get the name)
    : shp_type - shape type (Point, Polyline, Polygon, Multipoint, Multipatch)
    : - others: 'areaFieldName', 'baseName', 'catalogPath','featureType',
    :   'fields', 'hasOID', 'hasM', 'hasZ', 'path'
    : - all_flds =[i.name for i in desc['fields']]
    """
    desc = arcpy.da.Describe(in_fc)
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
    :  with ID, X, Y values. Optionally convert to points, otherwise centroid.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    a.dtype = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    del cur
    return a


def _ndarray(in_fc, to_pnts=True, flds=None, SR=None):
    """Convert featureclass geometry (in_fc) to a structured ndarray including
    :  options to select fields and specify a spatial reference.
    :
    :Requires:
    :--------
    : in_fc - input featureclass
    : to_pnts - True, convert the shape to points. False, centroid returned.
    : flds - '*' for all, others: 'Shape',  ['SHAPE@X', 'SHAPE@Y'], or specify
    """
    if flds is None:
        flds = "*"
    if SR is None:
        desc = arcpy.da.Describe(in_fc)
        SR = desc['spatialReference']
    args = [in_fc, flds, None, SR, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    del cur
    return a


def _two_arrays(in_fc, both=True, split=True):
    """Send to a numpy structured/array and split it into a geometry array
    :  and an attribute array.  They can be joined back later if needed.
    :
    :Note:  The geometry array is returned as an object array.  See the
    :----   main documentation
    :
    :Requires:
    :--------
    :functions:
    :  _xyID - function to get geometry array
    :  _ndarray - function to get the x, y, id array and attribute array
    :   fc_info(in_fc) - function needed to return fc properties
    :parameters:
    :  both  - True, to return both arrays, False to return just geometry
    :  split - True, split points by their geometry groups as an object array
    :         - False, a sequential array with shape = (N,)
    :variables:
    :  dt_a = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    :  dt_b = [('IDs', '<i4'), ('Xc', '<f8'), ('Yc', '<f8')]
    :  dt_b.extend(b.dtype.descr[2:])
    :       extend the dtype using the attribute dtype minus geometry and id
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
        b = _ndarray(in_fc, to_pnts=False)
        dt_b.extend(b.dtype.descr[2:])
        b.dtype = dt_b
    return a, b


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
