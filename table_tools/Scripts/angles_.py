# -*- coding: utf-8 -*-
"""
angles_
=======

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

def angles_poly(a, inside=True, in_deg=True, kind="sum"):
    """Sequential angles from a poly* shape
	NOTE: this line.... angle = np.sum(angles)
	      can be changed to `np.min`, `np.max` or others
		  depending on what needs to be returned
    """
    import numpy as np
    a = a.getPart()
    a =np.asarray([[i.X, i.Y] for j in a for i in j])
    if len(a) < 2:
        return None
    elif len(a) == 2:  # **** check
        ba = a[1] - a[0]
        return np.arctan2(*ba[::-1])
    else:
        angles = []
        if np.allclose(a[0], a[-1]):  # closed loop
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
            if not np.allclose(ang, np.pi):  # check for extra vertices
                angles.append(ang)
    if in_deg:
        angles = np.degrees(angles)
    if kind == "sum":
        angle = np.sum(angles)
    elif kind == "min":
        angle = np.min(angles)
    elif kind == "max":
        angle = np.max(angles)
    return angle
#__esri_field_calculator_splitter__
#angles_poly(!Shape!)