# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
: https://github.com/jsmolka/hull/blob/master/hull.py
: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
:        line-segments-intersect#565282
: http://www.codeproject.com/Tips/862988/Find-the-intersection-
:       point-of-two-line-segments
: considerCollinearOverlapAsIntersect => co_check
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import math
import numpy as np
from fc import _xy
from tools import uniq
from geom import e_dist
from apt import output_polylines
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

PI = np.pi

# ---- Modified code from references
#
def nearest_n(pnts, pt, n):   # renamed tonearest_n(pnts, pt, n):
    """n-nearest neighbours for a pnts list.
    : pnts - points array (xs, ys)
    : pt - reference point (px, py)
    : n - nearest neighbours
    : srted => sorted(sqrt((xs[0] - px[0])**2 + (ys[1] - py[1])**2))[0:k]
    :Notes:
    :------
    :  for two 2d vectors U = (Ux, Uy) V = (Vx, Vy)
    :  the crossproduct is    U x V = Ux*Vy - Uy*Vx
    """
    nn_idx = np.argsort(e_dist(pt, pnts))[:n]
    return pnts[nn_idx]


def intersect_pnt(p0, p1, p2, p3):
    """Returns the point of intersection of the segment passing through two
    :  line segments (p0, p1) and (p2, p3)
    :Notes:
    :------
    :         p0,            p1,             p2,            p3
    : (array([0, 0]), array([10, 10]),array([0, 5]), array([5, 0]))
    : s: array([[ 0,  0],    h: array([[  0.,   0.,   1.],
    :           [10, 10],              [ 10.,  10.,   1.],
    :           [ 0,  5],              [  0.,   5.,   1.],
    :           [ 5,  0]])             [  5.,   0.,   1.]])
    :Reference:
    :---------
    : https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    """
    s = np.vstack([p0, p1, p2, p3])      # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])            # get first line
    l2 = np.cross(h[2], h[3])            # get second line
    x, y, z = np.cross(l1, l2)           # point of intersection
    if z == 0:                           # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


def pnt_in_list(pnt, pnts_list):
    """Check to see if a point is in a list of points
    """
    is_in = np.any([np.isclose(pnt, i) for i in pnts_list])
    return is_in


def intersects(*args):
    """Line intersection check.  Two lines or 4 points that form the lines.
    :Requires:
    :--------
    :  intersects(line0, line1) or intersects(p0, p1, p2, p3)
    :   p0, p1 -> line 1
    :   p2, p3 -> line 2
    :Returns: boolean, if the segments do intersect
    :--------
    :References:
    :--------
    : https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
    :        line-segments-intersect#565282
    """
    if len(args) == 2:
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    # ---- First check ----   np.cross(p1-p0, p3-p2 )
    s10_x, s10_y = p1 - p0
    s32_x, s32_y = p3 - p2
    denom = s10_x * s32_y - s32_x * s10_y
    if np.isclose(denom, 0.0):
        return False
    #
    # ---- Second check ----  np.cross(p1-p0, p0-p2 )
    den_gt0 = denom > 0
    s02_x, s02_y = p0 - p2
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == den_gt0:
        return False
    #
    # ---- Third check ----  np.cross(p3-p2, p0-p2)
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == den_gt0:
        return False
    #
    if ((s_numer > denom) == den_gt0) or ((t_numer > denom) == den_gt0):
        return False
    #
    # ---- check to see if the intersection point is one of the input points
    t = t_numer / denom
    x0, y0 = p0           # substitute p0 in the equation
    x = x0 + (t * s10_x)
    y = y0 + (t * s10_y)
    if np.any([np.all([x, y] == i) for i in [p0, p1, p2, p3]]):
        return False
    return True


def angle(p0, p1, prv_ang=0):
    """Angle between two points and the previous angle, or zero.
    """
#   PI = np.pi
    a0 = (np.arctan2(p0[1] - p1[1], p0[0] - p1[0]) - prv_ang)
    a0 = a0 % (PI * 2) - PI
    return a0


def pnt_in_poly(pnt, poly):
    """Point is in polygon. ## fix this and use pip from arraytools
    """
    poly = np.array(poly)
    size = len(poly)
    xs = poly[:, 0]
    ys = poly[:, 1]
    x, y = pnt
    for i in range(size):
        min_ = min([xs[i], poly[(i + 1) % size][0]])
        max_ = max([xs[i], poly[(i + 1) % size][0]])
        if min_ < x <= max_:
            p = ys[i] - poly[(i + 1) % size][1]
            q = xs[i] - poly[(i + 1) % size][0]
            point_y = (x - xs[i]) * p / q + ys[i]
            if point_y < y:
                return True
    return False


def concave(in_pnts, k):
    """Calculates the concave hull for given points
    :Requires:
    :--------
    : in_pnts - initially the input set of points with duplicates removes and
    :    sorted on the Y value first, lowest Y at the top (?)
    : k - initially the number of points to start forming the concave hull,
    :    k will be the initial set of neighbors
    :Notes:  This recursively calls itself to check concave hull
    :-----
    """
    k = max(k, 3)  # Make sure k >= 3
    pnts_ = in_pnts  # list(set(in_pnts[:]))  # Remove duplicates

    if len(pnts_) < 3:
        raise Exception("Dataset length cannot be smaller than 3")
    elif len(pnts_) == 3:
        return pnts_  # Points are a polygon already
    k = min(k, len(pnts_) - 1)  # Make sure k neighbours can be found

    # ---- get the point with the minimum y, throw it into the hull
    # remove it from the input list
    # first_pnt = cur_pnt = min(dataset, key=lambda x: x[1])
    cur_pnt = pnts_[np.argmin(pnts_, axis=0)[1]]  #
    first_pnt = cur_pnt
    hull = [first_pnt]  # Initialize hull with first point
    pnts_ = np.delete(pnts_, 0, axis=0)  # Remove first point from dataset
    prv_ang = 0
    # ----  need np.all since curr_pnt and first_pnt are arrays
    while (np.all(cur_pnt != first_pnt) or len(hull) == 1) and len(pnts_) > 0:
        if len(hull) == 3:  # Add first point again
            pnts_ = np.append(pnts_, np.atleast_2d(first_pnt), axis=0)
        nn_pnts = nearest_n(pnts_, cur_pnt, k)  # Find nearest neighbours
#        print("nn_pnts\n{}".format(nn_pnts))
        #
        c_points = sorted(nn_pnts, key=lambda x: -angle(x, cur_pnt, prv_ang))
        #
        is_True = True
        i = -1
        while is_True and i < len(c_points) - 1:
            i += 1
            last_point = 1 if np.all(c_points[i] == first_pnt) else 0
            j = 1
            is_True = False
            while not is_True and j < len(hull) - last_point:
                args = [hull[-1], c_points[i], hull[-j - 1], hull[-j]]
                is_True = intersects(*args)
                j += 1
        if is_True:  # All intersect, try with higher number of neighbours
            return concave(in_pnts, k + 1)
        prv_ang = angle(c_points[i], cur_pnt)
        cur_pnt = c_points[i]
        hull.append(cur_pnt)  # Valid candidate was found
#        pnts_.remove(cur_pnt)  # check
        whr = np.where(pnts_ == cur_pnt)[0]
        pnts_ = np.delete(pnts_, whr, axis=0)
    # ---- final check again
    for point in pnts_:  # final point in polygon check
        if not pnt_in_poly(point, hull):
            return concave(in_pnts, k + 1)
    #
    return hull

# ------- original ----------------------------------------------------------
#
#def np_perp(a):
#    """perpendicular"""
#    b = np.empty_like(a)
#    b[0] = a[1]
#    b[1] = -a[0]
#    return b
#
#
#def np_cross(a, b):
#    """Cross product"""
#    return np.dot(a, np_perp(b))
#
#def knn(points, p, k):
#    """
#    Calculates k nearest neighbours for a given point.
#
#    :param points: list of points
#    :param p: reference point
#    :param k: amount of neighbours
#    :return: list
#    """
#    return sorted(points, key=lambda x: math.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2))[0:k]
#
#
#def np_seg_intersect(a, b, co_check = False):
#    """Line segment intersections
#    : https://stackoverflow.com/questions/563198/how-do-you-detect-where-
#    :       two-line-segments-intersect/565282#565282
#    : http://www.codeproject.com/Tips/862988/Find-the-intersection-
#    :       point-of-two-line-segments
#    : considerCollinearOverlapAsIntersect => co_check
#    """
#    def np_perp(a):
#        """perpendicular"""
#        b = np.empty_like(a)
#        b[0] = a[1]
#        b[1] = -a[0]
#        return b
#
#    def np_cross(a, b):
#        """Cross product"""
#        return np.dot(a, np_perp(b))
#
#    r = a[1] - a[0]
#    s = b[1] - b[0]
#    v = b[0] - a[0]
#    numer = np_cross(v, r)
#    denom = np_cross(r, s)
#    # If r x s = 0 and (q - p) x r = 0, then the two lines are collinear.
#    if np.isclose(denom, 0) and np.isclose(numer, 0):
#        # 1. If either  0 <= (q - p) * r <= r * r or 0 <= (p - q) * s <= * s
#        # then the two lines are overlapping,
#        if(co_check):
#            vDotR = np.dot(v, r)
#            aDotS = np.dot(-v, s)
#            chk0 = (0 <= vDotR and vDotR <= np.dot(r, r))
#            chk1 = (0 <= aDotS and aDotS <= np.dot(s, s))
#            if chk0 or chk1:
#                return True
#        # 2. If neither
#        #    0 <= (q - p) * r = r * r nor 0 <= (p - q) * s <= s * s
#        # then the two lines are collinear but disjoint.
#        # No need to implement this expression, it follows from
#        # the expression above.
#        return None
#    if np.isclose(denom, 0.0) and not np.isclose(numer, 0.0):
#        # Parallel and non intersecting
#        return None
#    u = numer / denom
#    t = np_cross(v, s) / denom
#    if u >= 0 and u <= 1 and t >= 0 and t <= 1:
#        res = b[0] + (s*u)
#        return res
#    # Otherwise, the two line segments are not parallel but do not intersect.
#    return None
#def intersects(p1, p2, p3, p4):
#    """
#    Checks if lines p1, p2 and p3, p4 intersect.
#
#    :param p1, p2: line
#    :param p3, p4: line
#    :return: bool
#    """
#    p0_x, p0_y = p1
#    p1_x, p1_y = p2
#    p2_x, p2_y = p3
#    p3_x, p3_y = p4
#
#    s10_x = p1_x - p0_x
#    s10_y = p1_y - p0_y
#    s32_x = p3_x - p2_x
#    s32_y = p3_y - p2_y
#
#    denom = s10_x * s32_y - s32_x * s10_y
#    if denom == 0:
#        return False
#
#    denom_positive = denom > 0
#    s02_x = p0_x - p2_x
#    s02_y = p0_y - p2_y
#    s_numer = s10_x * s02_y - s10_y * s02_x
#    if (s_numer < 0) == denom_positive:
#        return False
#
#    t_numer = s32_x * s02_y - s32_y * s02_x
#    if (t_numer < 0) == denom_positive:
#        return False
#
#    if ((s_numer > denom) == denom_positive) or ((t_numer > denom) == denom_positive):
#        return False
#
#    t = t_numer / denom
#    x = p0_x + (t * s10_x)
#    y = p0_y + (t * s10_y)
#
#    if (x, y) in [p1, p2, p3, p4]:
#        return False
#
#    return True
#
#
#def angle(p1, p2, previous_angle=0):
#    """
#    Calculates angle between two points and previous angle.
#
#    :param p1: point
#    :param p2: point
#    :param previous_angle: previous angle
#    :return: float
#    """
#    return (math.atan2(p1[1] - p2[1], p1[0] - p2[0]) - previous_angle) % (math.pi * 2) - math.pi
#
#
#def point_in_polygon(point, polygon):
#    """
#    """
#    size = len(polygon)
#    for i in range(size):
#        min_ = min([polygon[i][0], polygon[(i + 1) % size][0]])
#        max_ = max([polygon[i][0], polygon[(i + 1) % size][0]])
#        if min_ < point[0] <= max_:
#            p = polygon[i][1] - polygon[(i + 1) % size][1]
#            q = polygon[i][0] - polygon[(i + 1) % size][0]
#            point_y = (point[0] - polygon[i][0]) * p / q + polygon[i][1]
#            if point_y < point[1]:
#                return True
#    return False
#
#
#def concave(points, k):
#    """
#    Calculates the concave hull for given points
#    Input is a list of 2D points [(x, y), ...]
#    k defines the number of of considered neighbours
#
#    :param points: list of points
#    :param k: considered neighbours
#    :return: list
#    """
#    k = max(k, 3)  # Make sure k >= 3
#    dataset = list(set(points[:]))  # Remove duplicates
#    if len(dataset) < 3:
#        raise Exception("Dataset length cannot be smaller than 3")
#    elif len(dataset) == 3:
#        return dataset  # Points are a polygon already
#    k = min(k, len(dataset) - 1)  # Make sure k neighbours can be found
#
#    first_point = current_point = min(dataset, key=lambda x: x[1])
#    hull = [first_point]  # Initialize hull with first point
#    dataset.remove(first_point)  # Remove first point from dataset
#    previous_angle = 0
#
#    while (current_point != first_point or len(hull) == 1) and len(dataset) > 0:
#        if len(hull) == 3:
#            dataset.append(first_point)  # Add first point again
#        kn_points = knn(dataset, current_point, k)  # Find nearest neighbours
#        c_points = sorted(kn_points, key=lambda x: -angle(x, current_point, previous_angle))
#
#        is_True = True
#        i = -1
#        while is_True and i < len(c_points) - 1:
#            i += 1
#            last_point = 1 if c_points[i] == first_point else 0
#            j = 1
#            is_True = False
#            while not is_True and j < len(hull) - last_point:
#                is_True = intersects(hull[-1], c_points[i], hull[-j - 1], hull[-j])
#                j += 1
#        if is_True:  # All points intersect, try again with higher number of neighbours
#            return concave(points, k + 1)
#        previous_angle = angle(c_points[i], current_point)
#        current_point = c_points[i]
#        hull.append(current_point)  # Valid candidate was found
#        dataset.remove(current_point)
#
#    for point in dataset:
#        if not point_in_polygon(point, hull):
#            return concave(points, k + 1)
#
#    return hull
#

# --------------------------------------------------------------------------
#def cross(o, a, b):
#    """
#    Calculates cross product.
#
#    :param o, a: vector
#    :param o, b: vector
#    :return: int
#    """
#    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
#
#
#def convex(points):
#    """
#    Calculates the convex hull for given points
#    Input is a list of 2D points [(x, y), ...]
#
#    :param points: list of points
#    :return: list
#    """
#    points = sorted(set(points))  # Remove duplicates
#    if len(points) <= 1:
#        return points
#
#    # Build lower hull
#    lower = []
#    for p in points:
#        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
#            lower.pop()
#        lower.append(p)
#
#    # Build upper hull
#    upper = []
#    for p in reversed(points):
#        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
#            upper.pop()
#        upper.append(p)
#
#    return lower[:-1] + upper[:-1]
# ----------------------------------------------------------------------------
def test():
    """ """
    a = [(207, 184), (393, 60), (197, 158), (197, 114), (128, 261),
          (442, 40), (237, 159), (338, 75), (194, 93), (33, 159),
          (393, 152), (433, 267), (324, 141), (384, 183), (273, 165),
          (250, 257), (423, 198), (227, 68), (120, 184), (214, 49),
          (256, 75), (379, 93), (312, 49), (471, 187), (366, 122)]
    a = np.asarray(a)
    idx_cr = np.lexsort((a[:, 0], a[:, 1]))
    in_pnts = np.asarray([a[i] for i in idx_cr])
    cx = concave(in_pnts, 3)
    return cx


def test_main():
    """ """
    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\r_sorted"
    out_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\r_convex_hull"
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    flds = ['SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, True, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = _xy(in_fc)                    # ---- get the points
    a = uniq(a, axis=0)               # ---- get the unique points
#    a_s = a[a[:, 1].argsort(axis=0)]  # sort to get lowest y-value
    idx_cr = np.lexsort((a[:, 0], a[:, 1]))
    in_pnts = np.asarray([a[i] for i in idx_cr])  # sort by column 1, then 0
#    output_polylines(out_fc, SR, [pl])  # ***** it works


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    from tools import uniq
#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\r_sorted"
#    out_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\r_convex_hull"
#    desc = arcpy.da.Describe(in_fc)
#    SR = desc['spatialReference']
#    flds = ['SHAPE@X', 'SHAPE@Y']
#    args = [in_fc, flds, None, None, True, (None, None)]
#    cur = arcpy.da.SearchCursor(*args)
#    a = _xy(in_fc)                    # ---- get the points
#    a = uniq(a, axis=0)               # ---- get the unique points
##    a_s = a[a[:, 1].argsort(axis=0)]  # sort to get lowest y-value
#    idx_cr = np.lexsort((a[:, 0], a[:, 1]))
#    in_pnts = np.asarray([a[i] for i in idx_cr])  # sort by column 1, then 0
#    output_polylines(out_fc, SR, [pl])  # ***** it works

#    in_pnts = [tuple(i) for i in a_s.tolist()]
#    pnts = concave(in_pnts, 3)
#    output_polylines(out_fc, SR, [pnts])
#    ps = [(207, 184), (393, 60), (197, 158), (197, 114), (128, 261),
#          (442, 40), (237, 159), (338, 75), (194, 93), (33, 159),
#          (393, 152), (433, 267), (324, 141), (384, 183), (273, 165),
#          (250, 257), (423, 198), (227, 68), (120, 184), (214, 49),
#          (256, 75), (379, 93), (312, 49), (471, 187), (366, 122)]
#    a = np.array(ps)
#    a_s = a[a[:, 1].argsort(axis=0)]  # sort to get lowest y-value
#    idx_cr = np.lexsort((a[:, 0], a[:, 1]))
#    in_pnts = np.asarray([a[i] for i in idx_cr])  # sort by column 1, then 0
##    cx = concave(in_pnts, 3)
