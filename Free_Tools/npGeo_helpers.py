# -*- coding: utf-8 -*-
"""
=============
npGeo_helpers
=============

Script :
    npGeo_helpers.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-07-16

Purpose :
    A numpy geometry class, its properties and methods.  These methods work
    with Geo arrays or np.ndarrays.  In the case of the former, the methods may
    be being called from Geo methods in such things as a list comprehension.

Notes:

References:

"""
# pylint: disable=R0904  # pylint issue
# pylint: disable=C0103  # invalid-name
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0201  # attribute defined outside __init__... none in numpy
# pylint: disable=W0621  # redefining name

import sys
import numpy as np
# from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
from scipy.spatial import ConvexHull as CH

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_angles_', '_area_centroid_', '_area_part_', '_ch_', '_ch_scipy',
           '_ch_simple_', '_nan_split_', '_o_ring_', '_pnts_on_line_',
           '_polys_to_segments_', '_polys_to_unique_pnts_',
           '_simplify_lines_']

# ===== Workers with Geo and ndarrays. =======================================


def _o_ring_(arr):
    """Collect the outer ring of a shape.  An outer ring is separated from
    its inner ring, a hole, by a ``null_pnt``.  Each shape is examined for
    these and the outer ring is split off for each part of the shape.
    Called by::
        angles, outer_rings, is_convex and convex_hulls
    """
    nan_check = np.isnan(arr[:, 0])
    if np.any(nan_check):  # split at first nan to do outer
        w = np.where(np.isnan(arr[:, 0]))[0]
        arr = np.split(arr, w)[0]
    return arr


def _area_part_(a):
    """Mini e_area, used by areas and centroids"""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.nansum((e0 - e1)*0.5)


def _area_centroid_(a):
    """Calculate area and centroid for a singlepart polygon, `a`.  This is also
    used to calculate area and centroid for a Geo array's parts.
    """
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    area = np.nansum((e0 - e1)*0.5)
    t = e1 - e0
    area = np.nansum((e0 - e1)*0.5)
    x_c = np.nansum((x1 + x0) * t, axis=0) / (area * 6.0)
    y_c = np.nansum((y1 + y0) * t, axis=0) / (area * 6.0)
    return area, np.asarray([-x_c, -y_c])


def _angles_(a, inside=True, in_deg=True):
    """Worker for Geo.angles. sequential points, a, b, c.

    Parameters
    ----------
    inside : boolean
        True, for interior angles.
    in_deg : boolean
        True for degrees, False for radians
    """
    #
    a = _o_ring_(a)             # work with the outer rings only
    dx, dy = a[0] - a[-1]
    if np.allclose(dx, dy):     # closed loop, remove duplicate
        a = a[:-1]
    ba = a - np.roll(a, 1, 0)   # just as fastish as concatenate
    bc = a - np.roll(a, -1, 0)  # but defitely cleaner
    cr = np.cross(ba, bc)
    dt = np.einsum('ij,ij->i', ba, bc)
    ang = np.arctan2(cr, dt)
    TwoPI = np.pi*2.
    if inside:
        angles = np.where(ang < 0, ang + TwoPI, ang)
    else:
        angles = np.where(ang > 0, TwoPI - ang, ang)
    if in_deg:
        angles = np.degrees(angles)
    return angles


def _ch_scipy(points):
    """Convex hull using scipy.spatial.ConvexHull. Remove null_pnts, calculate
    the hull, derive the vertices and reorder clockwise.
    """
    p_nonan = points[~np.isnan(points[:, 0])]
    out = CH(p_nonan)
    return out.points[out.vertices][::-1]


def _ch_simple_(in_points):
    """Calculates the convex hull for given points.  Removes null_pnts, finds
    the unique points, then determines the hull from the remaining
    """
    def _x_(o, a, b):
        """Cross-product for vectors o-a and o-b"""
        xo, yo = o
        xa, ya = a
        xb, yb = b
        return (xa - xo)*(yb - yo) - (ya - yo)*(xb - xo)
    # ----
    points = in_points[~np.isnan(in_points[:, 0])]
    _, idx = np.unique(points, return_index=True, axis=0)
    points = points[idx]
    if len(points) <= 3:
        return in_points
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and _x_(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _x_(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    ch = np.array(lower[:-1] + upper)[::-1]  # sort clockwise
    if np.all(ch[0] != ch[-1]):
        ch = np.vstack((ch, ch[0]))
    return ch


def _ch_(points, threshold=50):
    """Perform a convex hull using either simple methods or scipy's"""
    points = points[~np.isnan(points[:, 0])]
    if len(points) > threshold:
        return _ch_scipy(points)
    return _ch_simple_(points)


def _pnts_on_line_(a, spacing=1):  # densify by distance
    """Add points, at a fixed spacing, to an array representing a line.
    **See**  ``densify_by_distance`` for documentation.

    Parameters
    ----------
    a : array
        A sequence of `points`, x,y pairs, representing the bounds of a polygon
        or polyline object
    spacing : number
        Spacing between the points to be added to the line.
    """
    N = len(a) - 1                                    # segments
    dxdy = a[1:, :] - a[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    steps = leng/spacing                              # step distance
    deltas = dxdy/(steps.reshape(-1, 1))              # coordinate steps
    pnts = np.empty((N,), dtype='O')                  # construct an `O` array
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1, -1)        # add the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


def _polys_to_unique_pnts_(a, as_structured=True):
    """Derived from polys_to_points, but allowing for recreation of original
    point order and unique points.  NaN's are removed.
    """
    good = a[~np.isnan(a.X)]
    uni, idx, _, cnts = np.unique(good, True, True,
                                  return_counts=True, axis=0)
    if as_structured:
        N = uni.shape[0]
        dt = [('New_ID', '<i4'), ('Xs', '<f8'), ('Ys', '<f8'), ('Num', '<i4')]
        z = np.zeros((N,), dtype=dt)
        z['New_ID'] = idx
        z['Xs'] = uni[:, 0]
        z['Ys'] = uni[:, 1]
        z['Num'] = cnts
        return z[np.argsort(z, order='New_ID')]  # dump nan coordinates
    return np.asarray(uni)


def _polys_to_segments_(a, as_2d=True, as_structured=False):
    """Segment poly* structures into o-d pairs from start to finish

    Parameters
    ----------
    a: array
        A 2D array of x,y coordinates representing polyline or polygons.
    as_2d : boolean
        Returns a 2D array of from-to point pairs, [xf, yf, xt, yt] if True.
        If False, they are returned as a 3D array in the form
        [[xf, yf], [xt, yt]]
    as_structured : boolean
        Optional structured/recarray output.  Field names are currently fixed.

    Notes
    -----
    Any row containing np.nan is removed since this would indicate that the
    shape contains the null_pnt separator.
    prn_tbl`` if you want to see a well formatted output.
    """
    s0, s1 = a.shape
    fr_to = np.zeros((s0-1, s1 * 2), dtype=a.dtype)
    fr_to[:, :2] = a[:-1]
    fr_to[:, 2:] = a[1:]
    fr_to = fr_to[~np.any(np.isnan(fr_to), axis=1)]
    if as_structured:
        dt = np.dtype([('X_orig', 'f8'), ('Y_orig', 'f8'),
                       ('X_dest', 'f8'), ('Y_dest', 'f8')])
        return uts(fr_to, dtype=dt)
    if not as_2d:
        s0, s1 = fr_to.shape
        return fr_to.reshape(s0, s1//2, s1//2)
    return fr_to


def _simplify_lines_(a, deviation=10):
    """Simplify array
    """
    ang = _angles_(a, inside=True, in_deg=True)
    idx = (np.abs(ang - 180.) >= deviation)
    sub = a[1: -1]
    p = sub[idx]
    return a, p, ang


# ===========================================================================
#  Keep???
def _nan_split_(arr):
    """Split at an array with nan values for an  ndarray."""
    s = np.isnan(arr[:, 0])                 # nan is used to split the 2D arr
    if np.any(s):
        w = np.where(s)[0]
        ss = np.split(arr, w)
        subs = [ss[0]]                      # collect the first full section
        subs.extend(i[1:] for i in ss[1:])  # slice off nan from the remaining
        return np.asarray(subs)
    return arr


# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
#
# testing = True
# if len(sys.argv) == 1 and testing:
#     pth = r"C:\Git_Dan\npgeom\data\Polygons.geojson"
#     #pth = r"C:\Git_Dan\npgeom\data\Oddities.geojson"
#     #pth = r"C:\Git_Dan\npgeom\data\Ontario_LCConic.geojson"
#     msg = "\nRunning : {}\npath to geojson : {}".format(script, pth)
# else:
#     msg = "\n{}\npath to geojson : {}".format(script, None)
#
# print(msg)

# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # optional controls here
