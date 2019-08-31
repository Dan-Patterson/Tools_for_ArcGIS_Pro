# -*- coding: utf-8 -*-
"""
=============
npg_geom
=============

Script :
    npg_geom.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-08-26

Purpose :
    A numpy geometry class, its properties and methods.  These methods work
    with Geo arrays or np.ndarrays.  In the case of the former, the methods may
    be being called from Geo methods in such things as a list comprehension.

Notes:

flatten a searchcursor to points and/or None

>>> in_fc = "C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
>>> SR = npg.getSR(in_fc)
>>> c = arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR)
>>> pnts = [[[[p for p in arr] for arr in r[1]]] for r in c]
>>> c.reset()  # don't forget to reset the cursor

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
# from numpy.lib.recfunctions import repack_fields
from numpy.lib.recfunctions import _keep_fields
from numpy.lib.recfunctions import append_fields
from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay

import npg_io
# from npGeo_io import fc_data

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['_angles_', '_area_centroid_', '_area_part_', '_ch_', '_ch_scipy_',
           '_ch_simple_', '_nan_split_', '_o_ring_', '_pnts_on_line_',
           '_densify_by_dist_',
           '_polys_to_segments_', '_polys_to_unique_pnts_',
           '_simplify_lines_']


# ===== Workers with Geo and ndarrays. =======================================
# ---- area and centroid helpers
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


def _area_part_(a):
    """Mini e_area, used by areas and centroids"""
    x0, y1 = (a.T)[:, 1:]
    x1, y0 = (a.T)[:, :-1]
    e0 = np.einsum('...i,...i->...i', x0, y0)
    e1 = np.einsum('...i,...i->...i', x1, y1)
    return np.nansum((e0 - e1)*0.5)


# ---- angle helpers
#
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

    Requires
    --------
    Called by ``pnt_on_poly``.
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


def _densify_by_dist_(a, spacing=1):
    """Call _pnts_on_line_"""
    return _pnts_on_line_(a, spacing)


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
# ---- points in or on geometries --------------------------------------------
#
def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points

    sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in


def pnt_on_seg(pnt, seg):
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


def pnt_on_poly(pnt, poly):
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
    e_dist is represented by _e_2d and pnt_on_seg by its equivalent below.

    _line_dir_ is from it's equivalent line_dir included here.

    This may be as simple as finding the closest point on the edge, but if
    needed, an orthogonal projection onto a polygon/line edge will be done.
    This situation arises when the distance to two sequential points is the
    same.
    """
    def _e_2d_(a, p):
        """ array points to point distance... mini e_dist"""
        diff = a - p[np.newaxis, :]
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
    if np.all(poly[0] == poly[-1]):  # strip off any duplicate
        poly = poly[:-1]
    # ---- determine the distances
    d = _e_2d_(poly, pnt)   # abbreviated edist =>  d = e_dist(poly, pnt)
    key = np.argsort(d)[0]  # dist = d[key]
    if key == 0:
        seg = np.vstack((poly[-1:], poly[:3]))
    elif (key + 1) >= len(poly):
        seg = np.vstack((poly[-2:], poly[:1]))
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
        r = (pnt[0], pnt[1], n1[0], n1[1], np.asscalar(d1), np.asscalar(ang))
        return r
    else:
        dest = [n2[0], n2[1]]
        ang = _line_dir_(pnt, dest)
        ang = np.mod((450.0 - ang), 360.)
        r = (pnt[0], pnt[1], n2[0], n2[1], np.asscalar(d2), np.asscalar(ang))
        return r


def p_o_p(pnts, poly):
    """ main runner to run multiple points to a polygon
    """
    result = []
    for p in pnts:
        result.append(pnt_on_poly(p, poly))
    result = np.asarray(result)
    dt = [('X0', '<f8'), ('Y0', '<f8'), ('X1', '<f8'), ('Y1', '<f8'),
          ('Dist', '<f8'), ('Angle', '<f8')]
    z = np.zeros((len(result),), dtype=dt)
    names = z.dtype.names
    for i, n in enumerate(names):
        z[n] = result[:, i]
    return z


def point_in_polygon(pnt, poly):  # pnt_in_poly(pnt, poly):  #
    """Point is in polygon. ## fix this and use pip from arraytools
    """
    x, y = pnt
    N = len(poly)
    for i in range(N):
        x0, y0, xy = [poly[i][0], poly[i][1], poly[(i + 1) % N]]
        c_min = min([x0, xy[0]])
        c_max = max([x0, xy[0]])
        if c_min < x <= c_max:
            p = y0 - xy[1]
            q = x0 - xy[0]
            y_cal = (x - x0) * p / q + y0
            if y_cal < y:
                return True
    return False


# ==== Attribute tools ======================================================
# (1) from featureclass table
def table_crosstab(in_tbl, flds=None):
    """Derive the unique attributes in a table for all or selected fields.

    Parameters
    ----------
    in_tbl : table
        A featureclass or its table.
    flds : fields
        If None, then all fields in the table are used.
        Make sure that you do not include sequential id fields or all table
        records will be returned.

    Notes
    -----
    None or <null> values in tables are converted to proper nodata values
    depending on the field type.  This is handled by the call to fc_data which
    uses _make_nulls_ to do the work.
    """
    a = npg_io.fc_data(in_tbl)
    if flds is None:
        flds = list(a.dtype.names)
    uni, idx, cnt = np.unique(a[flds], True, False, True)
    out_arr = append_fields(uni, "Counts", cnt, usemask=False)
    return out_arr


# (2) from 2 numpy ndarrays
def rc_crosstab(row, col, reclassed=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Parameters
    ----------
    row, col : text
        row and column array/field

    Returns
    -------
    ctab : the crosstabulation result as row, col, count array
    rc_ : similar to above, but the row/col unique pairs are combined.
    """
    dt = np.dtype([('row', row.dtype), ('col', col.dtype)])
    rc_zip = list(zip(row, col))
    rc = np.asarray(rc_zip, dtype=dt)
    u, idx, cnts = np.unique(rc, return_index=True, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u['row'], u['col'], cnts)), dtype=rcc_dt)
    # ----
    if reclassed:
        rc2 = np.array(["{}_{}".format(*i) for i in rc_zip])
        u2, idx2, cnts2 = np.unique(rc2, return_index=True, return_counts=True)
        dt = [('r_c', u2.dtype.str), ('cnts', '<i4')]
        rc_ = np.array(list(zip(u2, cnts2)), dtype=dt)
        return rc_
    return ctab


# (3) from a structured array
def array_crosstab(a, flds=None):
    """Frequency and crosstabulation for structured arrays.

    Parameters
    ----------
    a : array
       input structured array

    flds : string or list
       Fields/columns to use in the analysis.  For a single column, a string
       is all that is needed.  Multiple columns require a list of field names.

    Notes
    -----
    (1) slice the input array by the classification fields
    (2) sort the sliced array using the flds as sorting keys
    (3) use unique on the sorted array to return the results
    (4) reassemble the original columns and the new count data
    """
    if flds is None:
        return None
    if isinstance(flds, (str)):
        flds = [flds]
    # a = repack_fields(a[flds])  # need to repack fields
    a = _keep_fields(a, flds)
    idx = np.argsort(a, axis=0, order=flds)  # (2) sort
    a_sort = a[idx]
    uniq, counts = np.unique(a_sort, return_counts=True)  # (3) unique, count
    dt = uniq.dtype.descr
    dt.append(('Count', '<i4'))
    fr = np.empty_like(uniq, dtype=dt)
    names = fr.dtype.names
    vals = list(zip(*uniq)) + [counts.tolist()]  # (4) reassemble
    N = len(names)
    for i in range(N):
        fr[names[i]] = vals[i]
    return fr


"""
Demo

r = np.array(['A', 'A', 'B', 'B', 'B', 'A', 'A', 'C', 'C', 'A'], dtype='<U1')
c = np.array(['b', 'a', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'a'], dtype='<U1')
rc = np.array(["{}_{}".format(*i) for i in zip(r, c)])
u, idx, cnts = np.unique(rc, return_index=True, return_counts=True)
dt = [('r_c', u.dtype.str), ('cnts', '<i4')]
ctab = np.array(list(zip(u, cnts)), dtype=dt)
"""


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
