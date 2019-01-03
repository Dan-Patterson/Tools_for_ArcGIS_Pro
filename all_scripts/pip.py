# -*- coding: UTF-8 -*-
"""
pip
===

Script :   pip.py

Author :   Dan.Patterson@carleton.ca

Modified : 2019-01-02

Purpose:
--------
  Incarnations of point in polygon searches.  Includes, points in extent and
  crossing number.

References:
----------

`<https://stackoverflow.com/questions/33051244/numpy-filter-points-within-
bounding-box/33051576#33051576>`_.

`<https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html>`_.  ** good

Notes:
------
Remove points that are outside of the polygon extent, then filter those
using the crossing number approach to test whether a point is within.

**Sample run**

>>> a, ext = array_demo()
>>> poly = extent_poly(ext)
>>> p0 = np.array([341999, 5021999])  # just outside
>>> p1 = np.mean(poly, axis=0)        # extent centroid
>>> pnts - 10,000 points within the full extent, 401 points within the polygon

(1) pnts_in_extent:

>>> %timeit pnts_in_extent(pnts, ext, in_out=False)
143 µs ± 2.16 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> %timeit pnts_in_extent(pnts, ext, in_out=True)
274 µs ± 9.12 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

(2) crossing_num with pnts_in_extent check (current version):

>>> %timeit crossing_num(pnts, poly)
9.68 ms ± 120 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

(3) pure crossing_num:

>>> %timeit crossing_num(pnts, poly)
369 ms ± 19.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ----10| ------20| ------30| ------40| ------50| ------60| ------70| ------80|
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

__all__ = ['extent_poly',
           'pnts_in_extent',
           'crossing_num']


def extent_poly(ext):
    """Construct the extent rectangle from the extent points which are the
    lower left and upper right points [LB, RT]
    """
    LB, RT = ext
    L, B = LB
    R, T = RT
    box = [LB, [L, T], RT, [R, B], LB]
    ext_rect = np.array(box)
    return ext_rect


def pnts_in_extent(pnts, ext, in_out=True):
    """Point(s) in polygon test using numpy and logical_and to find points
    within a box/extent.

    Requires:
    --------
    pnts : array
      an array of points, ndim at least 2D
    ext : numbers
      the extent of the rectangle being tested as an array of the left bottom
      (LB) and upper right (RT) coordinates
    in_out : boolean
      - True to return both the inside and outside points.
      - False for inside only.

    Notes:
    ------
    - comp : np.logical_and( great-eq LB, less RT)  condition check
    - inside : np.where(np.prod(comp, axis=1) == 1) if both true, product = 1
    - case : comp returns [True, False] so you take the product
    - idx_in : indices derived using where since case will be 0 or 1
    - inside : slice the pnts using idx_in
    """
    pnts = np.atleast_2d(pnts)  # account for single point
    outside = None
    LB, RT = ext
    comp = np.logical_and((LB <= pnts), (pnts <= RT))
    case = comp[..., 0] * comp[..., 1]
    idx_in = np.where(case)[0]
    inside = pnts[idx_in]
    if in_out:
        idx_out = np.where(~case)[0]  # invert case
        outside = pnts[idx_out]
    return inside, outside


def crossing_num(pnts, poly):
    """Points in polygon implementation of crossing number largely from pnpoly
    in its various incarnations.  This version also does a within extent
    test to pre-process the points, keeping those within the extent to be
    passed on to the crossing number section.

    Requires:
    ---------
    pnts_in_extent : function
      Method to limit the retained points to those within the polygon extent.
      See 'pnts_in_extent' for details
    pnts : array
      point array
    poly : polygon
      closed-loop as an array

    """
    xs = poly[:, 0]
    ys = poly[:, 1]
    dx = np.diff(xs)
    dy = np.diff(ys)
    ext = np.array([poly.min(axis=0), poly.max(axis=0)])
    inside, outside = pnts_in_extent(pnts, ext, in_out=False)
    is_in = []
    for pnt in inside:
        cn = 0    # the crossing number counter
        x, y = pnt
        for i in range(len(poly)-1):      # edge from V[i] to V[i+1]
            u = ys[i] <= y < ys[i+1]
            d = ys[i] >= y > ys[i+1]
            if np.logical_or(u, d):       # compute x-coordinate
                vt = (y - ys[i]) / dy[i]
                if x < (xs[i] + vt * dx[i]):
                    cn += 1
        is_in.append(cn % 2)  # either even or odd (0, 1)
    result = inside[np.nonzero(is_in)]
    return result


def _demo():
    """ used in the testing
    : polygon layers
    : C:/Git_Dan/a_Data/testdata.gdb/Carp_5x5km  full 25 polygons
    : C:/Git_Dan/a_Data/testdata.gdb/subpoly     centre polygon with 'ext'
    : C:/Git_Dan/a_Data/testdata.gdb/centre_4    above, but split into 4
    """
    ext = np.array([[342000, 5022000], [343000, 5023000]])
    in_fc = r'C:\Git_Dan\a_Data\testdata.gdb\xy_10k'
    SR = arcpy.SpatialReference(2951)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, ['SHAPE@X', 'SHAPE@Y'],
                                          spatial_reference=SR)
    pnts = a.view(dtype=np.float).reshape(len(a), 2)
    poly = extent_poly(ext)
    p0 = np.array([341999., 5021999.])
    p1 = np.mean(poly, axis=0)
    pnts = np.array([p0, p1])
    return pnts, ext, poly, p0, p1


def _demo1():
    """Simple check for known points"""
    ext = np.array([[342000, 5022000], [343000, 5023000]])
    poly = extent_poly(ext)
    p0 = np.array([341999, 5021999])
    p1 = np.mean(poly, axis=0)
    pnts = np.array([p0, p1])
    return pnts, ext, poly, p0, p1


if __name__ == "__main__":
    """Make some points for testing, create an extent, time each option.
    :
    :Profiling functions
    : %load_ext line_profiler
    : %lprun -f pnts_in_extent pnts_in_extent(pnts, ext)  # -f means function
    """
#    pnts, ext, poly, p0, p1 = _demo1()
