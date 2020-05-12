# -*- coding: UTF-8 -*-
"""
rotate_pnts
===========

Script:  rotate_pnts.py

Author:  Dan.Patterson@carleton.ca

Modified: 2018-07-23

Notes:
-----
>>> arcpy.da.FeatureClassToNumPyArray(in_table, field_names, {where_clause},
                                      {spatial_reference}, {explode_to_points},
                                      {skip_nulls}, {null_value})
>>> arcpy.da.NumPyArrayToFeatureClass(in_array, out_table, shape_fields,
                                      {spatial_reference})

Data references for standalone testing is from the Point_tools database
"""
import sys
import numpy as np
import arcpy
from arcpytools_pnt import fc_info, tweet

script = sys.argv[0]

arcpy.env.overwriteOutput = True

def extent_(a):
    """Extent of an array.

    Returns:
    --------
    L(eft), B(ottom), R(ight), T(op)

    >>> a = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    >>> extent_(a)
    [0, 0, 1, 1]
    """
    L, B = np.min(a, axis=0)
    R, T = np.max(a, axis=0)
    return [L, B, R, T]


def trans_rot(a, angle=0.0, unique=True):
    """Translate and rotate and array of points about the point cloud origin.

    Requires:
    ---------
    a : array
        2d array of x,y coordinates.
    angle : double
        angle in degrees in the range -180. to 180
    unique :
        If True, then duplicate points are removed.  If False, then this would
        be similar to doing a weighting on the points based on location.

    Returns:
    --------
    Points rotated about the origin and translated back.

    >>> a = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    >>> b = trans_rot(b, 45)
    >>> b
    array([[ 0.5,  1.5],
           [ 1.5,  1.5],
           [ 0.5, -0.5],
           [ 1.5, -0.5]])

    Notes:
    ------
    - if the points represent a polygon, make sure that the duplicate
    - np.einsum('ij,kj->ik', a - cent, R)  =  np.dot(a - cent, R.T).T
    - ik does the rotation in einsum

    >>> R = np.array(((c, s), (-s,  c)))  # clockwise about the origin
    """
    if unique:
        a = np.unique(a, axis=0)
    cent = a.mean(axis=0)
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s,  c)))
    return  np.einsum('ij,kj->ik', a - cent, R) + cent


# ---- main section ----
#
if len(sys.argv) == 1:
    testing = True
    fc = "/Point_tools.gdb/std_dist_center"
    flder = "/".join(script.split("/")[:-2])
    in_fc = flder + fc
    angle = 30.0
    out_fc = flder + "/Point_tools.gdb/rot_std_dist"

else:
    testing = False
    in_fc = sys.argv[1]
    angle = float(sys.argv[2])
    out_fc = sys.argv[3]


# ---- convert to array, shift and return ----
# Apparently, there can be problems writing directly to a featureclass
# so, write to in_memory changing the required field names, then copy out
#
shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
arr = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR, True)
a = arr[shp_fld]
new_pnts = trans_rot(a, angle)
nms = ['Feat_id', 'XYs'] + [i for i in arr.dtype.names[2:]]
arr.dtype.names = nms
arr['XYs'] = new_pnts
if not testing:
    arcpy.da.NumPyArrayToFeatureClass(arr, out_fc, ['XYs'])
#
msg = """
-------------------------------------
Input points..... {}
Rotation angle... {}
Output points.... {}
-------------------------------------
"""
tweet(msg.format(in_fc, angle, out_fc))
# ---- the end ----
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    fc = "/Point_tools.gdb/std_dist_center"
    flder = "/".join(script.split("/")[:-2])
    in_fc = flder + fc