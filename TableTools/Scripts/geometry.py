# -*- coding: utf-8 -*-
"""
geometry
========

Requires:
---------

This function is called by poly_field_calcs.py from the Table Tools toolset
It is designed to be imported and used in the field calculator

There is a check for redundant vertices which affects the sum and max angles.
Specifically, 3 points on a 2 point line will have 180 degree max

Useage:
------
In the calling script use

>>> from angles import angles_poly

In the field calculator calling function, feed it the shape field
    angles_poly(!Shape!)
"""

import numpy as np
import math
from arcpytools import fc_info, tweet
import arcpy

def angles_(a, in_deg=True, kind="sum"):
    """Sequential angles from a poly* shape

    `a` - shape
        A shape from the shape field in a geodatabase

    """
    import numpy as np
    a = a.getPart()
    a = np.asarray([[i.X, i.Y] for j in a for i in j])
    if len(a) < 2:
        return None
    elif len(a) == 2:
        ba = a[1] - a[0]
        return np.arctan2(*ba[::-1])
    a0 = a[0:-2]
    a1 = a[1:-1]
    a2 = a[2:]
    ba = a1 - a0
    bc = a1 - a2
    cr = np.cross(ba, bc)
    dt = np.einsum('ij,ij->i', ba, bc)
#    ang = np.arctan2(np.linalg.norm(cr), dt)
    ang = np.arctan2(cr, dt)
    two_pi = np.pi*2.
    angles = np.where(ang<0, ang + two_pi, ang)
    if in_deg:
        angles = np.degrees(angles)
    if in_deg:
        angles = np.degrees(angles)
    if kind == "sum":
        angle = np.sum(angles)
    elif kind == "min":
        angle = np.min(angles)
    elif kind == "max":
        angle = np.max(angles)
    return angle


def lengths_(a, kind="avg"):
    """poly* side lengths.
    Options include "min length", "max length", "avg length"
    """
    import numpy as np
    a = a.getPart()
    a = np.asarray([[i.X, i.Y] for j in a for i in j])
    if np.allclose(a[0], a[-1]):  # closed loop
        a = a[:-1]
    a0 = a[:-1]
    b0 = a[1:]
    diff = b0 - a0
    s = np.power(diff, 2)
    d = np.sqrt(s[:, 0] + s[:, 1])
    if kind == "avg":
        leng = np.mean(d)
    elif kind == "min":
        leng = np.min(d)
    elif kind == "max":
        leng = np.max(d)
    return leng

#__esri_field_calculator_splitter__
#angles_poly(!Shape!)


def to_array(in_fc):
    """Extract the shapes and produce a coordinate array.
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    in_flds = [oid_fld] + ['SHAPE@X', 'SHAPE@Y']
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, in_flds)
    a = a[['SHAPE@X', 'SHAPE@Y']]
    a = a.view(np.float64).reshape(a.shape[0], 2)
    return a

