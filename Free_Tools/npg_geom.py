# -*- coding: utf-8 -*-
"""
========
npg_geom
========

Script :
    npg_geom.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-09-06

Purpose :
    Geometry focused methods that work with Geo arrays or np.ndarrays.
    In the case of the former, the methods may be being called from Geo methods
    in such things as a list comprehension.

Notes
-----

How to flatten a searchcursor to points and/or None

>>> in_fc = "C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
>>> SR = npg.getSR(in_fc)
>>> with arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as c:
...     pnts = [[[[p for p in arr] for arr in r[1]]] for r in c]
>>> c.reset()  # don't forget to reset the cursor
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

# from numpy.lib.recfunctions import unstructured_to_structured as uts
# from numpy.lib.recfunctions import structured_to_unstructured as stu
# from numpy.lib.recfunctions import repack_fields

from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay

import npgeom as npg

# import npg_io
# from npGeo_io import fc_data

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_area_centroid_', '_angles_',
           '_ch_scipy_', '_ch_simple_', '_ch_',
           '_dist_along_', '_percent_along_', '_pnts_on_line_',
           '_polys_to_unique_pnts_',
           '_simplify_lines_',
           '_pnts_in_poly_', '_pnt_on_poly_', '_pnt_on_segment_', 'p_o_p',
           '_rotate_', '_tri_pnts_'
           ]


def extent_to_poly(extent, kind=2):
    """Create a polygon/polyline feature from an array of x,y values.  The
    array returned is ordered clockwise with the first and last point repeated
    to form a closed-loop.

    Parameters
    ----------
    extent : array-like
        The extent is specified as four float values in the form of
        L(eft), B(ottom), R(ight), T(op) eg. np.array([5, 5, 10, 10])
    kind : integer
        A value of 1 for a polyline, or 2 for a polygon.
    """
    if len(extent) != 4:
        print("Check the docs...\n{}".format(extent_to_poly.__doc__))
        return None
    L, B, R, T = extent
    L, R = min(L, R), max(L, R)
    B, T = min(B, T), max(B, T)
    ext = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
    return npg.Update_Geo([ext], K=kind)


# ===== Workers with Geo and ndarrays. =======================================
# ---- area and centroid helpers
def _area_centroid_(a):
    """Calculate area and centroid for a singlepart polygon, `a`.  This is also
    used to calculate area and centroid for a Geo array's parts.

    Notes
    -----
    For multipart shapes, just use this syntax:

    >>> # rectangle with hole
    >>> a0 = np.array([[[0., 0.], [0., 10.], [10., 10.], [10., 0.], [0., 0.]],
                      [[2., 2.], [8., 2.], [8., 8.], [2., 8.], [2., 2.]]])
    >>> [npg._area_centroid_(i) for i in a0]
    >>> [(100.0, array([ 5.00,  5.00])), (-36.0, array([ 5.00,  5.00]))]
    """
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    t = e1 - e0
    area = np.nansum((e0 - e1)*0.5)
    x_c = np.nansum((x1 + x0) * t, axis=0) / (area * 6.0)
    y_c = np.nansum((y1 + y0) * t, axis=0) / (area * 6.0)
    return area, np.asarray([-x_c, -y_c])


# ---- angle helper
#
def _angles_(a, inside=True, in_deg=True):
    """Worker for Geo.angles. sequential points, a, b, c for the first bit in
    a shape, so interior holes are removed in polygons and the first part of
    a multipart shape is used.  Use multipart_to_singlepart if you want to
    process that type.

    Parameters
    ----------
    inside : boolean
        True, for interior angles.
    in_deg : boolean
        True for degrees, False for radians
    """
    #
    dx, dy = a[0] - a[-1]
    if np.allclose(dx, dy):     # closed loop, remove duplicate
        a = a[:-1]
    ba = a - np.roll(a, 1, 0)   # just as fastish as concatenate
    bc = a - np.roll(a, -1, 0)  # but definitely cleaner
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


# ---- convex hull helpers
#
def _ch_scipy_(points):
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
        """Cross-product for vectors o-a and o-b... a<--o-->b"""
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
        ch = np.concatenate((ch, ch[0]), axis=0)  # np.vstack((ch, ch[0]))
    return ch


def _ch_(points, threshold=50):
    """Perform a convex hull using either simple methods or scipy's."""
    points = points[~np.isnan(points[:, 0])]
    if len(points) > threshold:
        return _ch_scipy_(points)
    return _ch_simple_(points)


# ---- distance, densification helpers
#
def _dist_along_(a, dist=0):
    """Add a point along a poly feature at a distance from the start point.

    Requires
    --------
    val : number
      `val` is assumed to be a value between 0 and to total length of the
      poly feature.  If <= 0, the first point is returned.  If >= total
      length the last point is returned.

    Notes
    -----
    Determine the segment lengths and the cumulative length.  From the latter,
    locate the desired distance relative to it and the indices of the start
    and end points.

    The coordinates of those points and the remaining distance is used to
    derive the location of the point on the line.

    See Also
    --------
    _percent_along : function
        Similar to this function but measures distance as a percentage.
    """
    dxdy = a[1:, :] - a[:-1, :]                        # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    cumleng = np.concatenate(([0], np.cumsum(leng)))   # cumulative length
    if dist <= 0:              # check for faulty distance or start point
        return a[0]
    if dist >= cumleng[-1]:    # check for greater distance than cumulative
        return a[-1]
    _end_ = np.digitize(dist, cumleng)
    x1, y1 = a[_end_]
    _start_ = _end_ - 1
    x0, y0 = a[_start_]
    t = (dist - cumleng[_start_]) / leng[_start_]
    xt = x0 * (1. - t) + (x1 * t)
    yt = y0 * (1. - t) + (y1 * t)
    return np.array([xt, yt])


def _percent_along_(a, percent=0):
    """Add a point along a poly feature at a distance from the start point.
    The distance is specified as a percentage of the total poly feature length.

    See Also
    --------
    _dist_along : function
        Similar to this function but measures distance as a finite value from
        the start point.

    Requires
    --------
    Called by ``pnt_on_poly``.
    """
    if percent > 1.:
        percent /= 100.
    dxdy = a[1:, :] - a[:-1, :]                        # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    cumleng = np.concatenate(([0], np.cumsum(leng)))
    perleng = cumleng / cumleng[-1]
    if percent <= 0:              # check for faulty distance or start point
        return a[0]
    if percent >= perleng[-1]:    # check for greater distance than cumulative
        return a[-1]
    _end_ = np.digitize(percent, perleng)
    x1, y1 = a[_end_]
    _start_ = _end_ - 1
    x0, y0 = a[_start_]
    t = (percent - perleng[_start_])
    xt = x0 * (1. - t) + (x1 * t)
    yt = y0 * (1. - t) + (y1 * t)
    return np.array([xt, yt])


def _pnts_on_line_(a, spacing=1, is_percent=False):  # densify by distance
    """Add points, at a fixed spacing, to an array representing a line.
    **See**  ``densify_by_distance`` for documentation.

    Parameters
    ----------
    a : array
        A sequence of `points`, x,y pairs, representing the bounds of a polygon
        or polyline object.
    spacing : number
        Spacing between the points to be added to the line.
    is_percent : boolean
        Express the densification as a percent of the total length.

    Requires
    --------
    Called by ``pnt_on_poly``.
    """
    N = len(a) - 1                                    # segments
    dxdy = a[1:, :] - a[:-1, :]                       # coordinate differences
    leng = np.sqrt(np.einsum('ij,ij->i', dxdy, dxdy))  # segment lengths
    if is_percent:                                    # as percentage
        spacing = abs(spacing)
        spacing = min(spacing/100, 1.)
        steps = (sum(leng) * spacing) / leng          # step distance
    else:
        steps = leng/spacing                          # step distance
    deltas = dxdy/(steps.reshape(-1, 1))              # coordinate steps
    pnts = np.empty((N,), dtype='O')                  # construct an `O` array
    for i in range(N):              # cycle through the segments and make
        num = np.arange(steps[i])   # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1, -1)        # add the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


# ---- poly conversion helpers
#
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


def _simplify_lines_(a, deviation=10):
    """Simplify array
    """
    ang = _angles_(a, inside=True, in_deg=True)
    idx = (np.abs(ang - 180.) >= deviation)
    sub = a[1: -1]
    p = sub[idx]
    return a, p, ang


# ---- points in or on geometries --------------------------------------------
#
def _pnts_in_poly_(pnts, poly):
    """Points in polygon, implemented using crossing number largely derived
    from **pnpoly** in its various incarnations.

    This version does a ``within extent`` test to pre-process the points.
    Points meeting this condition are passed on to the crossing number section.

    Parameters
    ----------
    pnts : array
        point array
    poly : polygon
        Closed-loop as an array.  The last and first point will be the same in
        a correctly formed polygon.

    Notes
    -----
    Helpers from From arraytools.pntinply.
    """
    def _in_ext_(pnts, ext):
        """ Returns the points within an extent."""
        LB, RT = ext
        comp = np.logical_and(LB < pnts, pnts <= RT)
        idx = np.logical_and(comp[..., 0], comp[..., 1])
        return pnts[idx]

    def _crossing_num_(pnts, poly):
        """The implementation of pnply."""
        xs = poly[:, 0]
        ys = poly[:, 1]
        dx = np.diff(xs)
        dy = np.diff(ys)
        ext = np.array([poly.min(axis=0), poly.max(axis=0)])
        inside = _in_ext_(pnts, ext)
        is_in = []
        for pnt in inside:
            cn = 0    # the crossing number counter
            x, y = pnt
            for i in range(len(poly)-1):      # edge from V[i] to V[i+1]
                if np.logical_or((ys[i] <= y < ys[i+1]),
                                 (ys[i] >= y > ys[i+1])):
                    vt = (y - ys[i]) / dy[i]  # compute x-coordinate
                    if x < (xs[i] + vt * dx[i]):
                        cn += 1
            is_in.append(cn % 2)  # either even or odd (0, 1)
        return inside[np.nonzero(is_in)]
    # ----
    inside = _crossing_num_(pnts, poly)
    return inside


def _pnt_on_poly_(pnt, poly):
    """Find closest point location on a polygon/polyline.

    See : ``p_o_p`` for batch running of multiple points to a polygon.

    Parameters
    ----------
    pnt : 1D ndarray array
        XY pair representing the point coordinates.
    poly : 2D ndarray array
        A sequence of XY pairs in clockwise order is expected.  The first and
        last points may or may not be duplicates, signifying sequence closure.

    Returns
    -------
    A list of [x, y, distance, angle] for the intersection point on the line.
    The angle is relative to north from the origin point to the point on the
    polygon.

    Notes
    -----
    ``e_dist`` is represented by _e_2d and pnt_on_seg by its equivalent below.

    ``_line_dir_`` is from it's equivalent line_dir included here.

    This may be as simple as finding the closest point on the edge, but if
    needed, an orthogonal projection onto a polygon/line edge will be done.
    This situation arises when the distance to two sequential points is the
    same.
    """
    def _e_2d_(a, p):
        """ array points to point distance... mini e_dist"""
        diff = a - p[None, :]
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    def _pnt_on_seg_(seg, pnt):
        """mini pnt_on_seg function normally required by pnt_on_poly"""
        x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
        dist_ = dx*dx + dy*dy  # squared length
        u = ((x0 - x1)*dx + (y0 - y1)*dy)/dist_
        u = max(min(u, 1), 0)  # u must be between 0 and 1
        xy = np.array([dx, dy])*u + [x1, y1]
        return xy

    def _line_dir_(orig, dest):
        """mini line direction function"""
        orig = np.atleast_2d(orig)
        dest = np.atleast_2d(dest)
        dxy = dest - orig
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        return ang
    #
    pnt = np.asarray(pnt)
    poly = np.asarray(poly)
    if np.all(poly[0] == poly[-1]):  # strip off any duplicate points
        poly = poly[:-1]
    # ---- determine the distances
    d = _e_2d_(poly, pnt)   # abbreviated edist =>  d = e_dist(poly, pnt)
    key = np.argsort(d)[0]  # dist = d[key]
    if key == 0:  # np.vstack((poly[-1:], poly[:3]))
        seg = np.concatenate((poly[-1:], poly[:3]), axis=0)
    elif (key + 1) >= len(poly):  # np.vstack((poly[-2:], poly[:1]))
        seg = np.concatenate((poly[-2:], poly[:1]), axis=0)
    else:
        seg = poly[key-1:key+2]       # grab the before and after closest
    n1 = _pnt_on_seg_(seg[:-1], pnt)  # abbreviated pnt_on_seg
    d1 = np.linalg.norm(n1 - pnt)
    n2 = _pnt_on_seg_(seg[1:], pnt)   # abbreviated pnt_on_seg
    d2 = np.linalg.norm(n2 - pnt)
    if d1 <= d2:
        dest = [n1[0], n1[1]]
        ang = _line_dir_(pnt, dest)
        ang = np.mod((450.0 - ang), 360.)
        r = (pnt[0], pnt[1], n1[0], n1[1], d1.item(), ang.item())
        return r
    dest = [n2[0], n2[1]]
    ang = _line_dir_(pnt, dest)
    ang = np.mod((450.0 - ang), 360.)
    r = (pnt[0], pnt[1], n2[0], n2[1], d2.item(), ang.item())
    return r


def _pnt_on_segment_(pnt, seg):
    """Orthogonal projection of a point onto a 2 point line segment.
    Returns the intersection point, if the point is between the segment end
    points, otherwise, it returns the distance to the closest endpoint.

    Parameters
    ----------
    pnt : array-like
        `x,y` coordinate pair as list or ndarray
    seg : array-like
        `from-to points`, of x,y coordinates as an ndarray or equivalent

    Notes
    -----
    >>> seg = np.array([[0, 0], [10, 10]])  # p0, p1
    >>> p = [10, 0]
    >>> pnt_on_seg(seg, p)
    array([5., 5.])

    Generically, with crossproducts and norms

    >>> d = np.linalg.norm(np.cross(p1-p0, p0-p))/np.linalg.norm(p1-p0)
    """
    x0, y0, x1, y1, dx, dy = *pnt, *seg[0], *(seg[1] - seg[0])
    dist_ = dx*dx + dy*dy  # squared length
    u = ((x0 - x1)*dx + (y0 - y1)*dy)/dist_
    u = max(min(u, 1), 0)
    xy = np.array([dx, dy])*u + [x1, y1]
    d = xy - pnt
    return xy, np.hypot(d[0], d[1])


def p_o_p(pnts, poly):
    """Main runner to run multiple points to a polygon.
    """
    result = []
    for p in pnts:
        result.append(_pnt_on_poly_(p, poly))
    result = np.asarray(result)
    dt = [('X0', '<f8'), ('Y0', '<f8'), ('X1', '<f8'), ('Y1', '<f8'),
          ('Dist', '<f8'), ('Angle', '<f8')]
    z = np.zeros((len(result),), dtype=dt)
    names = z.dtype.names
    for i, n in enumerate(names):
        z[n] = result[:, i]
    return z


# ---- rotate helper
def _rotate_(geo_arr, R, as_group, clockwise):
    """Rotation helper.

    Parameters
    ----------
    geo_arr : array
        The input geo array, which is split here.
    as_group : boolean
        False, rotated about the extent center.  True, rotated about each
        shapes' center.
    R : array
        The rotation matrix, passed on from Geo.rotate.
    clockwise : boolean
    """
    shapes = geo_arr.shapes
    out = []
    if as_group:
        uniqs = []
        for chunk in shapes:
            _, idx = np.unique(chunk, True, axis=0)
            uniqs.append(chunk[np.sort(idx)])
        cents = [np.nanmean(i, axis=0) for i in uniqs]
        for i, chunk in enumerate(shapes):
            ch = np.einsum('ij,jk->ik', chunk-cents[i], R) + cents[i]
            out.append(ch)
        return out
    cent = np.nanmean(geo_arr, axis=0)
    for chunk in shapes:
        ch = np.einsum('ij,jk->ik', chunk-cent, R) + cent
        out.append(ch)
    return out


# ---- triangulation, Delaunay helper
#
def _tri_pnts_(pnts):
    """Triangulate the points and return the triangles.

    Parameters:
    -----------
    pnts : array
        Points for a shape or a group of points in array format.
        Either geo.shapes or np.ndarray.
    out : array
        An array of triangle points.

    Notes:
    ------
    The simplices are ordered counterclockwise, this is reversed in this
    implementation.

    References
    ----------
    `<C:/Arc_projects/Polygon_lineTools/Scripts/triangulate.py>`_.
    """
    pnts = pnts[~np.isnan(pnts[:, 0])]  # strip nan points
    pnts = np.unique(pnts, axis=0)    # get the unique points only
    avg = np.mean(pnts, axis=0)
    p = pnts - avg
    tri = Delaunay(p)
    simps = tri.simplices
    # ---- indices holder, fill with indices, repeat first and roll CW
    # translate the points back
    z = np.zeros((len(simps), 4), dtype='int32')
    z[:, :3] = simps
    z[:, 3] = simps[:, 0]
    z = z[:, ::-1]
    new_pnts = p[z] + avg
    new_pnts = new_pnts.reshape(-1, 2)
    return new_pnts

# ---- Not included yet -----------------------------------------------------
#


# ===========================================================================
# Extras used elsewhere
'''
def _area_part_(a):
    """Mini e_area, used by areas and centroids"""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.nansum((e0 - e1)*0.5)


def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points.

    sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in

'''
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # optional controls here

"""
Demo

r = np.array(['A', 'A', 'B', 'B', 'B', 'A', 'A', 'C', 'C', 'A'], dtype='<U1')
c = np.array(['b', 'a', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'a'], dtype='<U1')
rc = np.array(["{}_{}".format(*i) for i in zip(r, c)])
u, idx, cnts = np.unique(rc, return_index=True, return_counts=True)
dt = [('r_c', u.dtype.str), ('cnts', '<i4')]
ctab = np.array(list(zip(u, cnts)), dtype=dt)
"""
