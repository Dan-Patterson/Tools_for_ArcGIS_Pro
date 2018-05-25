# -*- coding: UTF-8 -*-
"""
:Script:   hulls.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-03-10
:Purpose:  tools for working with numpy arrays
:
:References:
: https://community.esri.com/blogs/dan_patterson/2018/03/11/
:       concave-hulls-the-elusive-container
: https://github.com/jsmolka/hull/blob/master/hull.py
: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
:        line-segments-intersect#565282
: http://www.codeproject.com/Tips/862988/Find-the-intersection-
:       point-of-two-line-segments
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from arcpytools import tweet, output_polylines, output_polygons
import arcpy
import warnings
import math


warnings.simplefilter('ignore', FutureWarning)

arcpy.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

PI = math.pi


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
    p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = *p0, *p1, *p2, *p3
    s10_x = p1_x - p0_x
    s10_y = p1_y - p0_y
    s32_x = p3_x - p2_x
    s32_y = p3_y - p2_y
    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0.0:
        return False
    #
    # ---- Second check ----  np.cross(p1-p0, p0-p2 )
    den_gt0 = denom > 0
    s02_x = p0_x - p2_x
    s02_y = p0_y - p2_y
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
    # substitute p0 in the equation
    x = p0_x + (t * s10_x)
    y = p0_y + (t * s10_y)
    # be careful that you are comparing tuples to tuples, lists to lists
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        return False
    return True


def angle(p0, p1, prv_ang=0):
    """Angle between two points and the previous angle, or zero.
    """
    ang = math.atan2(p0[1] - p1[1], p0[0] - p1[0])
    a0 = (ang - prv_ang)
    a0 = a0 % (PI * 2) - PI
    return a0


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


def knn(pnts, p, k):
    """
    Calculates k nearest neighbours for a given point.

    :param points: list of points
    :param p: reference point
    :param k: amount of neighbours
    :return: list
    """
    s = sorted(pnts,
               key=lambda x: math.sqrt((x[0]-p[0])**2 + (x[1]-p[1])**2))[0:k]
    return s


def concave(points, k):
    """Calculates the concave hull for given points
    :Requires:
    :--------
    : points - initially the input set of points with duplicates removes and
    :    sorted on the Y value first, lowest Y at the top (?)
    : k - initially the number of points to start forming the concave hull,
    :    k will be the initial set of neighbors
    :Notes:  This recursively calls itself to check concave hull
    : p_set - The working copy of the input points
    :-----
    """
    k = max(k, 3)  # Make sure k >= 3
    p_set = list(set(points[:]))  # Remove duplicates if not done already
    if len(p_set) < 3:
        raise Exception("p_set length cannot be smaller than 3")
    elif len(p_set) == 3:
        return p_set  # Points are a polygon already
    k = min(k, len(p_set) - 1)  # Make sure k neighbours can be found

    frst_p = cur_p = min(p_set, key=lambda x: x[1])
    hull = [frst_p]  # Initialize hull with first point
    p_set.remove(frst_p)  # Remove first point from p_set
    prev_ang = 0

    while (cur_p != frst_p or len(hull) == 1) and len(p_set) > 0:
        if len(hull) == 3:
            p_set.append(frst_p)  # Add first point again
        knn_pnts = knn(p_set, cur_p, k)  # Find nearest neighbours
        cur_pnts = sorted(knn_pnts, key=lambda x: -angle(x, cur_p, prev_ang))

        its = True
        i = -1
        while its and i < len(cur_pnts) - 1:
            i += 1
            last_point = 1 if cur_pnts[i] == frst_p else 0
            j = 1
            its = False
            while not its and j < len(hull) - last_point:
                its = intersects(hull[-1], cur_pnts[i], hull[-j - 1], hull[-j])
                j += 1
        if its:  # All points intersect, try a higher number of neighbours
            return concave(points, k + 1)
        prev_ang = angle(cur_pnts[i], cur_p)
        cur_p = cur_pnts[i]
        hull.append(cur_p)  # Valid candidate was found
        p_set.remove(cur_p)

    for point in p_set:
        if not point_in_polygon(point, hull):
            return concave(points, k + 1)
    #
    return hull


# ---- convex hull ----------------------------------------------------------
#
def cross(o, a, b):
    """Cross-product for vectors o-a and o-b
    """
    xo, yo = o
    xa, ya = a
    xb, yb = b
    return (xa - xo)*(yb - yo) - (ya - yo)*(xb - xo)
#    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex(points):
    """Calculates the convex hull for given points
    :Input is a list of 2D points [(x, y), ...]
    """
    points = sorted(set(points))  # Remove duplicates
    if len(points) <= 1:
        return points
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    print("lower\n{}\nupper\n{}".format(lower, upper))
    return np.array(lower[:-1] + upper)  # upper[:-1]) # for open loop


# ----------------------------------------------------------------------
# .... running script or testing code section
def _tool():
    """run when script is from a tool
    """
    in_fc = sys.argv[1]
    group_by = str(sys.argv[2])
    k_factor = int(sys.argv[3])
    hull_type = str(sys.argv[4])
    out_type = str(sys.argv[5])
    out_fc = sys.argv[6]
    return in_fc, group_by, k_factor, hull_type, out_type, out_fc


gdb_pth = "/".join(script.split("/")[:-2]) + "/Data/Point_tools.gdb"

if len(sys.argv) == 1:
    testing = True
    in_fc = gdb_pth + r"/r_sorted"
    group_by = 'Group_'
    k_factor = 3
    hull_type = 'concave'  # 'convex'
    out_type = 'Polyline'
    out_fc = gdb_pth + r"/r_11"
else:
    testing = False
    in_fc, group_by, k_factor, hull_type, out_type, out_fc = _tool()

msg = """\n
-----------------------------------------------------------------------
---- Concave/convex hull ----
script    {}
Testing   {}
in_fc     {}
group_by  {}
k_factor  {}
hull_type {}
out_type  {}
out_fc    {}
-----------------------------------------------------------------------

"""
args = [script, testing, in_fc, group_by, k_factor,
        hull_type, out_type, out_fc]
tweet(msg.format(*args))

desc = arcpy.da.Describe(in_fc)
SR = desc['spatialReference']
#
# (1) ---- get the points
out_flds = ['OID@', 'SHAPE@X', 'SHAPE@Y'] + [group_by]
a = arcpy.da.FeatureClassToNumPyArray(in_fc, out_flds, "", SR, True)
#
# (2) ---- determine the unique groupings of the points
uniq, idx, rev = np.unique(a[group_by], True, True)
groups = [a[np.where(a[group_by] == i)[0]] for i in uniq]
#
# (3) ---- for each group, perform the concave hull
hulls = []
for i in range(0, len(groups)):
    p = groups[i]
    n = len(p)
    p = p[['SHAPE@X', 'SHAPE@Y']]
    p = p.view(np.float64).reshape(n, 2)
    #
    # ---- point preparation section ------------------------------------
    p = np.array(list(set([tuple(i) for i in p])))  # Remove duplicates
    idx_cr = np.lexsort((p[:, 0], p[:, 1]))         # indices of sorted array
    in_pnts = np.asarray([p[i] for i in idx_cr])    # p[idx_cr]  #
    in_pnts = in_pnts.tolist()
    in_pnts = [tuple(i) for i in in_pnts]
    if hull_type == 'concave':
        cx = np.array(concave(in_pnts, k_factor))  # requires a list of tuples
    else:
        cx = np.array(convex(in_pnts))
    hulls.append(cx.tolist())
    # ----
    #
if out_type == 'Polyline':
    output_polylines(out_fc, SR, [hulls])
elif out_type == 'Polygon':
    output_polygons(out_fc, SR, [hulls])
else:
    for i in hulls:
        print("Hulls\n{}".format(np.array(i)))
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
